from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.string import resolve_expr

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

from src.tasks.amp_loco.ampmotion_loader import MotionLoader
from src.tasks.amp_loco.mdp.terminations import DelayedTerminationManager

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class MotionResetManager:
    """Manages motion frame data and delayed-reset logic for AMP environments."""

    _instance: MotionResetManager | None = None

    def __init__(self) -> None:
        self.walk_run_frames: dict[str, dict[str, torch.Tensor]] = {}
        self.recovery_frames: dict[str, dict[str, torch.Tensor]] = {}

    @classmethod
    def get(cls) -> MotionResetManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init(
        self,
        env: ManagerBasedRlEnv,
        motion_dir: str,
        recovery_dir: str | None = None,
    ) -> None:
        if motion_dir in self.walk_run_frames:
            return

        loader = MotionLoader(
            motion_dir=motion_dir,
            tgt_body_indexes=[],
            tgt_anchor_indexes=0,
            feet_indexes=0,
            device=str(env.device),
            recovery_dir=recovery_dir,
        )

        self.walk_run_frames[motion_dir] = self._concat_frames(loader.motion_data)
        motion_count = self.walk_run_frames[motion_dir]["root_pos"].shape[0]
        print(f"[MotionResetManager] Loaded {len(loader.motion_data)} clips, {motion_count} frames from {motion_dir}")

        if loader.motion_data_recovery:
            self.recovery_frames[motion_dir] = self._concat_frames(loader.motion_data_recovery)
            recovery_count = self.recovery_frames[motion_dir]["root_pos"].shape[0]
            print(f"[MotionResetManager] Loaded {len(loader.motion_data_recovery)} recovery clips, {recovery_count} frames from {recovery_dir}")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        motion_dir: str,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        home_keyframe: EntityCfg.InitialStateCfg | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

        if len(env_ids) == 0:
            return

        # Split into delay envs and normal envs.
        delay_mask = self._get_delay_env_mask(env)
        if delay_mask is not None:
            is_delay = delay_mask[env_ids]
            delay_ids = env_ids[is_delay]
            normal_ids = env_ids[~is_delay]
        else:
            delay_ids = env_ids[:0]  # empty
            normal_ids = env_ids

        # Reset normal envs with a configured home pose or walk/run data.
        if len(normal_ids) > 0:
            if home_keyframe is not None:
                self._write_keyframe_state(
                    env, normal_ids, home_keyframe, asset_cfg
                )
            else:
                self._write_reset_state(
                    env, normal_ids, self.walk_run_frames[motion_dir], asset_cfg
                )

        # Reset delay envs with recovery data (fallback to walk/run if unavailable).
        if len(delay_ids) > 0:
            recovery = self.recovery_frames.get(motion_dir)
            frames = recovery if recovery is not None else self.walk_run_frames[motion_dir]
            self._write_reset_state(env, delay_ids, frames, asset_cfg)

    def _get_delay_env_mask(self, env: ManagerBasedRlEnv) -> torch.Tensor | None:
        """Get delay env mask from DelayedTerminationManager if installed."""
        tm = env.termination_manager
        if isinstance(tm, DelayedTerminationManager):
            return tm._delay_env_mask
        return None

    def _write_reset_state(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        frames: dict[str, torch.Tensor],
        asset_cfg: SceneEntityCfg,
    ) -> None:
        total_frames = frames["root_pos"].shape[0]
        num_reset = len(env_ids)
        idx = torch.randint(0, total_frames, (num_reset,), device=env.device)

        asset: Entity = env.scene[asset_cfg.name]

        # --- Root pose ---
        root_pos = frames["root_pos"][idx]
        root_quat = frames["root_quat"][idx]
        positions = env.scene.env_origins[env_ids].clone()

        # --- Key Fix for terrain ---
        terrain_z = positions[:, 2].clone()
        positions[:, 2] = terrain_z + root_pos[:, 2]

        root_pose = torch.cat([positions, root_quat], dim=-1)
        asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)

        # --- Root velocity ---
        root_vel = torch.cat([frames["root_lin_vel"][idx], frames["root_ang_vel"][idx]], dim=-1)
        asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

        # --- Joint state ---
        joint_pos = frames["joint_pos"][idx]
        joint_vel = frames["joint_vel"][idx]

        soft_joint_pos_limits = asset.data.soft_joint_pos_limits
        assert soft_joint_pos_limits is not None
        joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
        joint_pos_clamped = joint_pos[:, asset_cfg.joint_ids].clamp_(
            joint_pos_limits[..., 0], joint_pos_limits[..., 1]
        )

        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, list):
            joint_ids = torch.tensor(joint_ids, device=env.device)

        asset.write_joint_state_to_sim(
            joint_pos_clamped,
            joint_vel[:, asset_cfg.joint_ids],
            env_ids=env_ids,
            joint_ids=joint_ids,
        )

    def _write_keyframe_state(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        keyframe: EntityCfg.InitialStateCfg,
        asset_cfg: SceneEntityCfg,
    ) -> None:
        asset: Entity = env.scene[asset_cfg.name]
        num_reset = len(env_ids)

        root_pos = torch.tensor(
            keyframe.pos, dtype=torch.float, device=env.device
        ).repeat(num_reset, 1)
        root_pos += env.scene.env_origins[env_ids]
        root_quat = torch.tensor(
            keyframe.rot, dtype=torch.float, device=env.device
        ).repeat(num_reset, 1)
        root_pose = torch.cat([root_pos, root_quat], dim=-1)
        asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)

        root_lin_vel = torch.tensor(
            keyframe.lin_vel, dtype=torch.float, device=env.device
        ).repeat(num_reset, 1)
        root_ang_vel = torch.tensor(
            keyframe.ang_vel, dtype=torch.float, device=env.device
        ).repeat(num_reset, 1)
        root_vel = torch.cat([root_lin_vel, root_ang_vel], dim=-1)
        asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

        if keyframe.joint_pos is None:
            raise ValueError(
                "MotionResetManager home_keyframe requires explicit joint_pos."
            )

        joint_pos = torch.tensor(
            resolve_expr(keyframe.joint_pos, asset.joint_names, 0.0),
            dtype=torch.float,
            device=env.device,
        ).repeat(num_reset, 1)
        joint_vel = torch.tensor(
            resolve_expr(keyframe.joint_vel, asset.joint_names, 0.0),
            dtype=torch.float,
            device=env.device,
        ).repeat(num_reset, 1)

        soft_joint_pos_limits = asset.data.soft_joint_pos_limits
        assert soft_joint_pos_limits is not None
        joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
        joint_pos_selected = joint_pos[:, asset_cfg.joint_ids].clamp_(
            joint_pos_limits[..., 0], joint_pos_limits[..., 1]
        )

        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, list):
            joint_ids = torch.tensor(joint_ids, device=env.device)

        asset.write_joint_state_to_sim(
            joint_pos_selected,
            joint_vel[:, asset_cfg.joint_ids],
            env_ids=env_ids,
            joint_ids=joint_ids,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _concat_frames(motions: list[dict]) -> dict[str, torch.Tensor]:
        root_pos_list = []
        root_quat_list = []
        root_lin_vel_list = []
        root_ang_vel_list = []
        joint_pos_list = []
        joint_vel_list = []
        for motion in motions:
            root_pos_list.append(motion["body_pos_w"][:, 0, :])
            root_quat_list.append(motion["body_quat_w"][:, 0, :])
            root_lin_vel_list.append(motion["body_lin_vel_w"][:, 0, :])
            root_ang_vel_list.append(motion["body_ang_vel_w"][:, 0, :])
            joint_pos_list.append(motion["dof_pos"])
            joint_vel_list.append(motion["dof_vel"])
        return {
            "root_pos": torch.cat(root_pos_list, dim=0),
            "root_quat": torch.cat(root_quat_list, dim=0),
            "root_lin_vel": torch.cat(root_lin_vel_list, dim=0),
            "root_ang_vel": torch.cat(root_ang_vel_list, dim=0),
            "joint_pos": torch.cat(joint_pos_list, dim=0),
            "joint_vel": torch.cat(joint_vel_list, dim=0),
        }


class UpwardRecoveryAssistance:
    """Apply an annealed upward force while delayed environments recover."""

    def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset: Entity = env.scene[asset_cfg.name]
        self._body_ids = asset_cfg.body_ids
        if not isinstance(self._body_ids, list) or not self._body_ids:
            raise ValueError(
                "UpwardRecoveryAssistance requires one or more explicit body names."
            )

        self._num_envs = env.num_envs
        self._num_bodies = len(self._body_ids)
        self._device = env.device

        self._force_range = tuple(cfg.params["force_range"])
        force_min, force_max = self._force_range
        if force_min < 0.0 or force_max < force_min:
            raise ValueError(
                f"Invalid upward assistance force range: {self._force_range}"
            )

        self._anneal_steps = int(cfg.params["anneal_steps"])
        if self._anneal_steps <= 0:
            raise ValueError("anneal_steps must be positive.")

        self._debug_vis_enabled = bool(cfg.params.get("debug_vis", True))
        self._viz_scale = float(cfg.params.get("viz_scale", 0.002))
        self._viz_width = float(cfg.params.get("viz_width", 0.02))
        if self._viz_scale <= 0.0:
            raise ValueError("viz_scale must be positive.")
        if self._viz_width <= 0.0:
            raise ValueError("viz_width must be positive.")

        self._sampled_magnitude = torch.zeros(
            self._num_envs, device=self._device
        )
        self._applied_magnitude = torch.zeros_like(self._sampled_magnitude)
        self._forces = torch.zeros(
            (self._num_envs, self._num_bodies, 3), device=self._device
        )
        self._torques = torch.zeros_like(self._forces)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        force_range: tuple[float, float],
        anneal_steps: int,
        asset_cfg: SceneEntityCfg,
        debug_vis: bool = True,
        viz_scale: float = 0.002,
        viz_width: float = 0.02,
    ) -> None:
        del env_ids, force_range, anneal_steps, asset_cfg, debug_vis, viz_scale, viz_width

        tm = env.termination_manager
        if isinstance(tm, DelayedTerminationManager):
            active = tm._delay_env_mask & (tm._delay_counters > 0)
        else:
            active = torch.zeros(
                self._num_envs, dtype=torch.bool, device=self._device
            )

        progress = min(env.common_step_counter / self._anneal_steps, 1.0)
        scale = max(1.0 - progress, 0.0)
        self._applied_magnitude.copy_(
            self._sampled_magnitude * scale * active.float()
        )

        # The selected body receives a world-frame +Z force at its center of mass.
        # Rewrite all components every step so recovery completion clears the wrench.
        self._forces.zero_()
        self._torques.zero_()
        self._forces[:, :, 2] = self._applied_magnitude[:, None]
        self._asset.write_external_wrench_to_sim(
            self._forces,
            self._torques,
            body_ids=self._body_ids,
        )

        log = env.extras.setdefault("log", {})
        log["Curriculum/upward_assistance_scale"] = torch.tensor(
            scale, device=self._device
        )
        log["Metrics/upward_assistance_active_fraction"] = active.float().mean()
        log["Metrics/upward_assistance_force_mean"] = (
            self._applied_magnitude.mean()
        )

    def debug_vis(self, visualizer: DebugVisualizer) -> None:
        """Draw upward-force arrows at the selected body center of mass."""
        if not self._debug_vis_enabled:
            return

        env_indices = list(visualizer.get_env_indices(self._num_envs))
        if not env_indices:
            return

        env_ids = torch.as_tensor(env_indices, device=self._device, dtype=torch.long)
        body_positions = self._asset.data.body_com_pos_w[env_ids][:, self._body_ids]
        magnitudes = self._applied_magnitude[env_ids]
        body_positions_np = body_positions.detach().cpu().numpy()
        magnitudes_np = magnitudes.detach().cpu().numpy()

        for env_row, _env_idx in enumerate(env_indices):
            magnitude = float(magnitudes_np[env_row])
            if magnitude <= 1.0e-6:
                continue

            for body_row in range(self._num_bodies):
                start = body_positions_np[env_row, body_row]
                end = start + np.array(
                    [0.0, 0.0, magnitude * self._viz_scale], dtype=np.float32
                )
                visualizer.add_arrow(
                    start=start,
                    end=end,
                    color=(1.0, 0.2, 0.05, 0.9),
                    width=self._viz_width,
                )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)

        num_reset = self._num_envs if isinstance(env_ids, slice) else len(env_ids)
        force_min, force_max = self._force_range
        sampled = torch.empty(num_reset, device=self._device).uniform_(
            force_min, force_max
        )
        self._sampled_magnitude[env_ids] = sampled
        self._applied_magnitude[env_ids] = 0.0

        zeros = torch.zeros(
            (num_reset, self._num_bodies, 3), device=self._device
        )
        self._asset.write_external_wrench_to_sim(
            zeros,
            zeros,
            env_ids=env_ids,
            body_ids=self._body_ids,
        )


# ------------------------------------------------------------------
# Event callback wrappers (thin delegates to singleton)
# ------------------------------------------------------------------

def init_motion_loader(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    recovery_dir: str | None = None,
    delay_reset_env_ratio: float = 0.0,
    max_delay_steps: int = 0,
) -> None:
    """Startup event: load motion data and optionally install delayed termination."""
    MotionResetManager.get().init(
        env=env,
        motion_dir=motion_dir,
        recovery_dir=recovery_dir,
    )

    # Install DelayedTerminationManager if requested.
    num_delay = int(env.num_envs * delay_reset_env_ratio)
    if num_delay > 0 and max_delay_steps > 0:
        delay_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        delay_indices = torch.randperm(env.num_envs, device=env.device)[:num_delay]
        delay_mask[delay_indices] = True
        env.termination_manager = DelayedTerminationManager(
            base=env.termination_manager,
            delay_env_mask=delay_mask,
            max_delay_steps=max_delay_steps,
        )
        print(
            "[init_motion_loader] DelayedTerminationManager installed: "
            f"{num_delay}/{env.num_envs} envs, max_delay_steps={max_delay_steps}"
        )


def reset_from_motion_data(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    home_keyframe: EntityCfg.InitialStateCfg | None = None,
) -> None:
    """Reset event: reset envs from motion frames or a home keyframe."""
    MotionResetManager.get().reset(
        env=env,
        env_ids=env_ids,
        motion_dir=motion_dir,
        asset_cfg=asset_cfg,
        home_keyframe=home_keyframe,
    )
