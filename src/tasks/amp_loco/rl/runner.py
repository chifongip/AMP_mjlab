"""AMP on-policy runner with ONNX export, integrated with migrated rsl_rl v6."""

from __future__ import annotations

import inspect
import os

import torch
import wandb
from tensordict import TensorDict

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
    attach_metadata_to_onnx,
    get_base_metadata,
)
from rsl_rl.runners import AMPOnPolicyRunner as _MigratedAMPOnPolicyRunner


# ---------------------------------------------------------------------------
# Config adapter
# ---------------------------------------------------------------------------

def _adapt_amp_config(train_cfg: dict) -> dict:
    """Move root-level AMP params into ``cfg["algorithm"]`` for v6 AMPPPO.

    The mjlab ``RslRlAmpRunnerCfg`` dataclass stores AMP fields at the top
    level.  v6 ``AMPPPO.construct_algorithm()`` reads them from
    ``cfg["algorithm"]``.  This adapter moves them there and fixes
    ``obs_groups`` values from tuples to lists (``asdict()`` preserves
    tuple defaults).
    """
    alg = train_cfg.setdefault("algorithm", {})

    # AMP keys that belong inside the algorithm sub-dict
    _amp_keys = [
        "amp_reward_coef",
        "amp_motion_files",
        "amp_num_preload_transitions",
        "amp_task_reward_lerp",
        "amp_discr_hidden_dims",
        "min_normalized_std",
        "amp_body_names",
        "amp_anchor_name",
    ]
    for key in _amp_keys:
        if key in train_cfg:
            alg.setdefault(key, train_cfg.pop(key))

    # Fix obs_groups: dataclass default uses tuples but resolve_obs_groups
    # expects list values.
    if "obs_groups" in train_cfg:
        train_cfg["obs_groups"] = {
            k: list(v) for k, v in train_cfg["obs_groups"].items()
        }

    # Ensure algorithm sub-dict has keys the v6 Logger expects.
    alg.setdefault("rnd_cfg", None)
    alg.setdefault("symmetry_cfg", None)

    # Strip keys from actor/critic sub-dicts that MLPModel doesn't accept.
    # RslRlModelCfg has fields for CNN/RNN models that are not MLPModel kwargs.
    _model_strip_keys = {"cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers", "class_name"}
    for section in ("actor", "critic"):
        if section in train_cfg:
            for key in _model_strip_keys:
                train_cfg[section].pop(key, None)

    return train_cfg


# ---------------------------------------------------------------------------
# ONNX export helpers
# ---------------------------------------------------------------------------

class _OnnxPolicyWrapper(torch.nn.Module):
    """Wrap v6 MLPModel actor + obs normalizer for ONNX export.

    The v6 ``MLPModel.forward()`` expects a ``TensorDict``, but ONNX
    tracing needs a plain tensor interface.  This wrapper takes a plain
    ``obs`` tensor, applies the normalizer, wraps it in a minimal
    ``TensorDict``, and calls the actor in deterministic mode.
    """

    def __init__(self, actor, obs_normalizer=None, obs_key: str = "actor"):
        super().__init__()
        self.actor = actor
        self.obs_normalizer = obs_normalizer
        self.obs_key = obs_key

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer(obs)
        td = TensorDict(
            {self.obs_key: obs}, batch_size=obs.shape[0], device=obs.device
        )
        return self.actor(td, stochastic_output=False)


def _onnx_export_kwargs_single_file() -> dict:
    """Build kwargs that request single-file ONNX export across torch versions."""
    try:
        params = inspect.signature(torch.onnx.export).parameters
    except (TypeError, ValueError):
        return {}

    if "external_data" in params:
        return {"external_data": False}
    if "use_external_data_format" in params:
        return {"use_external_data_format": False}
    return {}


def _inline_external_onnx_data(onnx_path: str) -> None:
    """Merge external tensor data back into a single ONNX file if needed."""
    data_path = f"{onnx_path}.data"
    if not os.path.exists(data_path):
        return

    try:
        import onnx

        model = onnx.load(onnx_path, load_external_data=True)
        onnx.save_model(model, onnx_path, save_as_external_data=False)
        if os.path.exists(data_path):
            os.remove(data_path)
        print(f"[INFO]: Inlined external ONNX data into single file: {onnx_path}")
    except Exception as exc:
        print(f"[WARN]: Failed to inline ONNX external data for {onnx_path}: {exc}")


# ---------------------------------------------------------------------------
# Custom AMP runner
# ---------------------------------------------------------------------------

class AMPOnPolicyRunner(_MigratedAMPOnPolicyRunner):
    """AMP on-policy runner with ONNX export for deployment.

    Extends the migrated v6 ``AMPOnPolicyRunner`` from rsl_rl, adding:
    - Config adaptation from mjlab format to v6 format.
    - Automatic ONNX export on every save.
    """

    env: RslRlVecEnvWrapper

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
        train_cfg = _adapt_amp_config(train_cfg)
        super().__init__(env, train_cfg, log_dir, device)

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def _export_policy_to_onnx(self, path: str, filename: str = "policy.onnx"):
        """Export actor + obs normalizer to a single ONNX file.

        Uses the uncompiled actor (``self.alg._raw_actor``) and its
        internal ``obs_normalizer`` so the exported model accepts raw
        observations directly.
        """
        actor = self.alg._raw_actor  # MLPModel (bare, not compiled)
        obs_normalizer = getattr(actor, "obs_normalizer", None)

        # Determine input dimension from the normalizer's running mean
        if obs_normalizer is not None and hasattr(obs_normalizer, "mean"):
            obs_dim = obs_normalizer.mean.numel()
        else:
            # Fallback: use pre-computed obs_dim from MLPModel
            obs_dim = self.alg._raw_actor.obs_dim

        wrapper = _OnnxPolicyWrapper(actor, obs_normalizer, obs_key="actor")
        wrapper.to("cpu")
        wrapper.eval()

        dummy_input = torch.zeros(1, obs_dim)
        os.makedirs(path, exist_ok=True)
        torch.onnx.export(
            wrapper,
            dummy_input,
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
            **_onnx_export_kwargs_single_file(),
        )
        _inline_external_onnx_data(os.path.join(path, filename))

        # Restore actor and normalizer to training device after ONNX export.
        actor.to(self.device)
        if obs_normalizer is not None and hasattr(obs_normalizer, "_mean"):
            obs_normalizer.to(self.device)

    # ------------------------------------------------------------------
    # Save (with ONNX export)
    # ------------------------------------------------------------------

    def save(self, path: str, infos=None):
        super().save(path, infos)
        policy_path = path.split("model")[0]
        filename = "policy.onnx"
        self._export_policy_to_onnx(policy_path, filename)

        run_name: str = (
            wandb.run.name
            if self.logger.logger_type in ("wandb", "WandbLogWriter") and wandb.run
            else "local"
        )
        onnx_path = os.path.join(policy_path, filename)
        metadata = get_base_metadata(self.env.unwrapped, run_name)
        attach_metadata_to_onnx(onnx_path, metadata)
        _inline_external_onnx_data(onnx_path)

        if self.logger.logger_type in ("wandb", "WandbLogWriter"):
            wandb.save(
                policy_path + filename, base_path=os.path.dirname(policy_path)
            )
