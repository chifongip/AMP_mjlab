"""Evaluate a recovery policy from a fixed, unassisted set of motion frames."""

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from src.tasks.amp_loco.mdp.metrics import (
  RECOVERY_MAXIMUM_TILT,
  RECOVERY_MINIMUM_HEIGHT,
  RECOVERY_STABLE_STEPS,
  body_tilt_from_state,
  standing_mask_from_state,
)


@dataclass(frozen=True)
class EvaluationConfig:
  checkpoint_file: str
  task_id: str = "Unitree-G1-AMP-Recovery-Flat"
  num_envs: int = 4096
  seed: int = 42
  evaluation_steps: int = 250
  stable_steps: int = RECOVERY_STABLE_STEPS
  minimum_height: float = RECOVERY_MINIMUM_HEIGHT
  maximum_tilt_degrees: float = math.degrees(RECOVERY_MAXIMUM_TILT)
  minimum_success_rate: float = 0.9
  device: str | None = None
  output_file: str | None = None
  deterministic: bool = True


def evaluate(cfg: EvaluationConfig) -> dict[str, object]:
  if cfg.evaluation_steps <= 0:
    raise ValueError("evaluation_steps must be positive.")
  if not 0 < cfg.stable_steps <= cfg.evaluation_steps:
    raise ValueError("stable_steps must be in [1, evaluation_steps].")

  checkpoint = Path(cfg.checkpoint_file).expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

  if cfg.deterministic:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
  configure_torch_backends(
    allow_tf32=not cfg.deterministic,
    deterministic=cfg.deterministic,
  )
  if cfg.deterministic:
    torch.use_deterministic_algorithms(True, warn_only=True)

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(cfg.task_id, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  env_cfg.terminations = {}
  if cfg.deterministic:
    for sensor_cfg in env_cfg.scene.sensors:
      if getattr(sensor_cfg, "reduce", None) == "none":
        sensor_cfg.reduce = "maxforce"
  for name in (
    "foot_friction",
    "encoder_bias",
    "base_com",
    "randomize_terrain",
    "push_robot",
  ):
    env_cfg.events.pop(name, None)
  env_cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 1.0

  agent_cfg = load_rl_cfg(cfg.task_id)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  try:
    runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=device)

    robot = env.scene["robot"]
    torso_ids, _ = robot.find_bodies(("torso_link",))
    torso_id = torso_ids[0]
    maximum_tilt = torch.deg2rad(
      torch.tensor(cfg.maximum_tilt_degrees, device=device)
    )

    initial_state = torch.cat(
      (robot.data.root_link_pose_w, robot.data.joint_pos), dim=-1
    )
    initial_state_hash = hashlib.sha256(
      initial_state.detach().cpu().numpy().tobytes()
    ).hexdigest()

    obs = wrapped.get_observations()
    stable_count = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)
    final_height = torch.zeros(cfg.num_envs, device=device)
    final_tilt = torch.zeros(cfg.num_envs, device=device)

    with torch.inference_mode():
      for _ in range(cfg.evaluation_steps):
        actions = policy(obs)
        obs, _, _, _ = wrapped.step(actions)

        final_height = (
          robot.data.body_link_pos_w[:, 0, 2] - env.scene.env_origins[:, 2]
        )
        torso_quat = robot.data.body_link_quat_w[:, torso_id]
        standing = standing_mask_from_state(
          final_height,
          torso_quat,
          robot.data.gravity_vec_w,
          cfg.minimum_height,
          maximum_tilt,
        )
        final_tilt = body_tilt_from_state(
          torso_quat,
          robot.data.gravity_vec_w,
        )
        stable_count = torch.where(
          standing, stable_count + 1, torch.zeros_like(stable_count)
        )

    stable = stable_count >= cfg.stable_steps
    success_rate = float(stable.float().mean().item())
    result: dict[str, object] = {
      "checkpoint": str(checkpoint),
      "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
      "task_id": cfg.task_id,
      "seed": cfg.seed,
      "num_envs": cfg.num_envs,
      "evaluation_steps": cfg.evaluation_steps,
      "stable_steps": cfg.stable_steps,
      "minimum_height": cfg.minimum_height,
      "maximum_tilt_degrees": cfg.maximum_tilt_degrees,
      "initial_state_sha256": initial_state_hash,
      "success_rate": success_rate,
      "mean_final_root_height": float(final_height.mean().item()),
      "mean_final_torso_tilt_degrees": float(
        torch.rad2deg(final_tilt).mean().item()
      ),
      "passed": success_rate >= cfg.minimum_success_rate,
      "practical_determinism": cfg.deterministic,
      "bitwise_reproducibility_expected": False,
    }
  finally:
    wrapped.close()

  output_file = (
    Path(cfg.output_file).expanduser().resolve()
    if cfg.output_file is not None
    else checkpoint.parent / "evaluation" / f"{checkpoint.stem}_seed_{cfg.seed}.json"
  )
  output_file.parent.mkdir(parents=True, exist_ok=True)
  output_file.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  print(f"[INFO] Evaluation written to: {output_file}")
  return result


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  result = evaluate(tyro.cli(EvaluationConfig))
  if not result["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
