from __future__ import annotations

from dataclasses import dataclass
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
from src.tasks.amp_loco.mdp.metrics import standing_mask_from_state
from src.tasks.amp_loco.mdp.terminations import DelayedTerminationManager

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


@dataclass
class AssistanceCurriculum:
    """State machine for competence-gated recovery assistance."""

    scales: tuple[float, ...]
    advance_threshold: float
    advance_hold_evaluations: int
    rollback_threshold: float
    rollback_hold_evaluations: int
    stage: int = 0
    advance_count: int = 0
    rollback_count: int = 0

    def __post_init__(self) -> None:
        if not self.scales or self.scales[-1] != 0.0:
            raise ValueError("assistance scales must be non-empty and end at zero.")
        if any(a <= b for a, b in zip(self.scales, self.scales[1:])):
            raise ValueError("assistance scales must be strictly decreasing.")
        if not 0.0 <= self.rollback_threshold < self.advance_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= rollback < advance <= 1."
            )
        if self.advance_hold_evaluations <= 0 or self.rollback_hold_evaluations <= 0:
            raise ValueError("curriculum hold durations must be positive.")
        if not 0 <= self.stage < len(self.scales):
            raise ValueError("initial curriculum stage is out of range.")

    @property
    def scale(self) -> float:
        return self.scales[self.stage]

    def update(self, success: float) -> bool:
        """Update performance evidence and return whether the stage changed."""
        self.advance_count = (
            self.advance_count + 1 if success >= self.advance_threshold else 0
        )
        self.rollback_count = (
            self.rollback_count + 1 if success < self.rollback_threshold else 0
        )

        if self.stage > 0 and self.rollback_count >= self.rollback_hold_evaluations:
            self.stage -= 1
            self.advance_count = 0
            self.rollback_count = 0
            return True
        elif (
            self.stage < len(self.scales) - 1
            and self.advance_count >= self.advance_hold_evaluations
        ):
            self.stage += 1
            self.advance_count = 0
            self.rollback_count = 0
            return True
        return False


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
    """Apply competence-gated upward force while delayed environments recover."""

    def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
        self._env = env
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

        self._evaluation_interval_steps = int(
            cfg.params["evaluation_interval_steps"]
        )
        if self._evaluation_interval_steps <= 0:
            raise ValueError("evaluation_interval_steps must be positive.")
        self._success_ema_alpha = float(cfg.params["success_ema_alpha"])
        if not 0.0 < self._success_ema_alpha <= 1.0:
            raise ValueError("success_ema_alpha must be in (0, 1].")
        self._success_ema: float | None = None
        self._last_evaluation_step = 0
        self._last_attempt_success_rate: float | None = None
        self._last_attempt_failure_rate: float | None = None
        self._last_attempt_timeout_rate: float | None = None
        self._curriculum = AssistanceCurriculum(
            scales=tuple(float(value) for value in cfg.params["assistance_scales"]),
            advance_threshold=float(cfg.params["advance_threshold"]),
            advance_hold_evaluations=int(
                cfg.params["advance_hold_evaluations"]
            ),
            rollback_threshold=float(cfg.params["rollback_threshold"]),
            rollback_hold_evaluations=int(
                cfg.params["rollback_hold_evaluations"]
            ),
            stage=int(cfg.params.get("initial_stage", 0)),
        )

        self._minimum_height = float(cfg.params["minimum_height"])
        self._maximum_tilt = float(cfg.params["maximum_tilt"])
        self._stable_steps_required = int(cfg.params["stable_steps"])
        if self._stable_steps_required <= 0:
            raise ValueError("stable_steps must be positive.")
        self._minimum_completed_attempts = int(
            cfg.params["minimum_completed_attempts"]
        )
        if self._minimum_completed_attempts <= 0:
            raise ValueError("minimum_completed_attempts must be positive.")

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
        self._attempt_active = torch.zeros(
            self._num_envs, dtype=torch.bool, device=self._device
        )
        self._stable_counts = torch.zeros(
            self._num_envs, dtype=torch.long, device=self._device
        )
        self._attempt_steps = torch.zeros_like(self._stable_counts)
        self._pending_successes = torch.zeros(
            (), dtype=torch.long, device=self._device
        )
        self._pending_failures = torch.zeros_like(self._pending_successes)
        self._pending_timeouts = torch.zeros_like(self._pending_successes)
        self._completed_successes = torch.zeros_like(self._pending_successes)
        self._completed_failures = torch.zeros_like(self._pending_successes)
        self._completed_timeouts = torch.zeros_like(self._pending_successes)
        self._completed_success_steps = torch.zeros_like(self._pending_successes)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        force_range: tuple[float, float],
        assistance_scales: tuple[float, ...],
        evaluation_interval_steps: int,
        success_ema_alpha: float,
        advance_threshold: float,
        advance_hold_evaluations: int,
        rollback_threshold: float,
        rollback_hold_evaluations: int,
        minimum_height: float,
        maximum_tilt: float,
        stable_steps: int,
        minimum_completed_attempts: int,
        asset_cfg: SceneEntityCfg,
        initial_stage: int = 0,
        debug_vis: bool = True,
        viz_scale: float = 0.002,
        viz_width: float = 0.02,
    ) -> None:
        del (
            env_ids,
            force_range,
            assistance_scales,
            evaluation_interval_steps,
            success_ema_alpha,
            advance_threshold,
            advance_hold_evaluations,
            rollback_threshold,
            rollback_hold_evaluations,
            minimum_height,
            maximum_tilt,
            stable_steps,
            minimum_completed_attempts,
            asset_cfg,
            initial_stage,
            debug_vis,
            viz_scale,
            viz_width,
        )

        tm = env.termination_manager
        if isinstance(tm, DelayedTerminationManager):
            active = tm._delay_env_mask & (tm._delay_counters > 0)
        else:
            active = torch.zeros(
                self._num_envs, dtype=torch.bool, device=self._device
            )

        standing_occupancy = self._update_attempts(env, tm)
        if (
            env.common_step_counter - self._last_evaluation_step
            >= self._evaluation_interval_steps
        ):
            self._evaluate_curriculum(env.common_step_counter)

        scale = self._curriculum.scale
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
        log["Curriculum/upward_assistance_stage"] = torch.tensor(
            self._curriculum.stage, device=self._device
        )
        log["Metrics/recovery_standing_occupancy"] = standing_occupancy
        log["Metrics/recovery_attempt_success_rate"] = torch.tensor(
            self._last_attempt_success_rate or 0.0,
            device=self._device,
        )
        log["Metrics/recovery_attempt_success_rate_ema"] = torch.tensor(
            self._success_ema if self._success_ema is not None else 0.0,
            device=self._device,
        )
        log["Metrics/recovery_attempt_failure_rate"] = torch.tensor(
            self._last_attempt_failure_rate or 0.0,
            device=self._device,
        )
        log["Metrics/recovery_attempt_timeout_rate"] = torch.tensor(
            self._last_attempt_timeout_rate or 0.0,
            device=self._device,
        )
        log["Metrics/recovery_attempts_pending"] = self._pending_attempts()
        log["Metrics/recovery_attempts_completed"] = (
            self._completed_successes
            + self._completed_failures
            + self._completed_timeouts
        )
        log["Metrics/recovery_mean_steps_to_success"] = (
            self._mean_steps_to_success()
        )
        log["Metrics/upward_assistance_active_fraction"] = active.float().mean()
        log["Metrics/upward_assistance_force_mean"] = (
            self._applied_magnitude.mean()
        )

    def _standing_mask(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        root_height = (
            self._asset.data.body_link_pos_w[:, 0, 2]
            - env.scene.env_origins[:, 2]
        )
        body_quat_w = self._asset.data.body_link_quat_w[:, self._body_ids[0]]
        return standing_mask_from_state(
            root_height,
            body_quat_w,
            self._asset.data.gravity_vec_w,
            self._minimum_height,
            self._maximum_tilt,
        )

    def _update_attempts(
        self,
        env: ManagerBasedRlEnv,
        tm: DelayedTerminationManager | object,
    ) -> torch.Tensor:
        recovery_mask = getattr(tm, "_delay_env_mask", None)
        if not isinstance(recovery_mask, torch.Tensor):
            return torch.zeros((), device=self._device)

        standing = self._standing_mask(env)
        eligible = recovery_mask & self._attempt_active
        self._attempt_steps[eligible] += 1
        self._stable_counts[eligible & standing] += 1
        self._stable_counts[eligible & ~standing] = 0

        succeeded = eligible & (
            self._stable_counts >= self._stable_steps_required
        )
        successes = torch.sum(succeeded)
        success_steps = torch.sum(self._attempt_steps[succeeded])
        self._pending_successes.add_(successes)
        self._completed_successes.add_(successes)
        self._completed_success_steps.add_(success_steps)
        self._attempt_active[succeeded] = False
        self._stable_counts[succeeded] = 0

        mask = recovery_mask.float()
        return torch.sum(standing.float() * mask) / torch.clamp(
            torch.sum(mask), min=1.0
        )

    def _record_reset_outcomes(
        self,
        env_ids: torch.Tensor | slice,
    ) -> None:
        tm = self._env.termination_manager
        recovery_mask = getattr(tm, "_delay_env_mask", None)
        if not isinstance(recovery_mask, torch.Tensor):
            self._attempt_active[env_ids] = False
            self._stable_counts[env_ids] = 0
            self._attempt_steps[env_ids] = 0
            return

        old_active = self._attempt_active[env_ids]
        failure_buf = getattr(tm, "_delay_failure_buf", None)
        timeout_buf = getattr(tm, "_delay_timeout_buf", None)
        if isinstance(failure_buf, torch.Tensor):
            failures = torch.sum(old_active & failure_buf[env_ids])
            self._pending_failures.add_(failures)
            self._completed_failures.add_(failures)
            failure_buf[env_ids] = False
        if isinstance(timeout_buf, torch.Tensor):
            timeouts = torch.sum(old_active & timeout_buf[env_ids])
            self._pending_timeouts.add_(timeouts)
            self._completed_timeouts.add_(timeouts)
            timeout_buf[env_ids] = False

        self._attempt_active[env_ids] = recovery_mask[env_ids]
        self._stable_counts[env_ids] = 0
        self._attempt_steps[env_ids] = 0

    def _pending_attempts(self) -> torch.Tensor:
        return (
            self._pending_successes
            + self._pending_failures
            + self._pending_timeouts
        )

    def _evaluate_curriculum(self, step: int) -> bool:
        self._last_evaluation_step = step
        completed = int(self._pending_attempts().item())
        if completed < self._minimum_completed_attempts:
            return False

        successes = int(self._pending_successes.item())
        failures = int(self._pending_failures.item())
        timeouts = int(self._pending_timeouts.item())
        success_rate = successes / completed
        self._last_attempt_success_rate = success_rate
        self._last_attempt_failure_rate = failures / completed
        self._last_attempt_timeout_rate = timeouts / completed

        if self._success_ema is None:
            self._success_ema = success_rate
        else:
            alpha = self._success_ema_alpha
            self._success_ema = (
                alpha * success_rate + (1.0 - alpha) * self._success_ema
            )
        transitioned = self._curriculum.update(self._success_ema)
        if transitioned:
            self._success_ema = None

        self._pending_successes.zero_()
        self._pending_failures.zero_()
        self._pending_timeouts.zero_()
        return True

    def _mean_steps_to_success(self) -> torch.Tensor:
        return self._completed_success_steps.float() / torch.clamp(
            self._completed_successes.float(), min=1.0
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

    def state_dict(self) -> dict[str, float | int | None]:
        """Return global curriculum state for checkpoint persistence."""
        return {
            "version": 2,
            "stage": self._curriculum.stage,
            "advance_count": self._curriculum.advance_count,
            "rollback_count": self._curriculum.rollback_count,
            "success_ema": self._success_ema,
            "last_evaluation_step": self._last_evaluation_step,
            "last_attempt_success_rate": self._last_attempt_success_rate,
            "last_attempt_failure_rate": self._last_attempt_failure_rate,
            "last_attempt_timeout_rate": self._last_attempt_timeout_rate,
            "pending_successes": int(self._pending_successes.item()),
            "pending_failures": int(self._pending_failures.item()),
            "pending_timeouts": int(self._pending_timeouts.item()),
            "completed_successes": int(self._completed_successes.item()),
            "completed_failures": int(self._completed_failures.item()),
            "completed_timeouts": int(self._completed_timeouts.item()),
            "completed_success_steps": int(
                self._completed_success_steps.item()
            ),
        }

    def load_state_dict(self, state: dict[str, float | int | None]) -> None:
        """Restore global curriculum state from a training checkpoint."""
        version = int(state.get("version", 1))
        stage = int(state.get("stage", self._curriculum.stage))
        if not 0 <= stage < len(self._curriculum.scales):
            raise ValueError(f"checkpoint assistance stage is invalid: {stage}")
        self._curriculum.stage = stage
        if version >= 2:
            self._curriculum.advance_count = int(state.get("advance_count", 0))
            self._curriculum.rollback_count = int(state.get("rollback_count", 0))
            success_ema = state.get("success_ema")
            self._success_ema = (
                None if success_ema is None else float(success_ema)
            )
            self._last_evaluation_step = int(
                state.get("last_evaluation_step", self._last_evaluation_step)
            )
        else:
            # Version 1 used duration-biased standing occupancy as evidence.
            self._curriculum.advance_count = 0
            self._curriculum.rollback_count = 0
            self._success_ema = None
            self._last_evaluation_step = int(
                getattr(self._env, "common_step_counter", 0)
            )
        for attr, key in (
            ("_last_attempt_success_rate", "last_attempt_success_rate"),
            ("_last_attempt_failure_rate", "last_attempt_failure_rate"),
            ("_last_attempt_timeout_rate", "last_attempt_timeout_rate"),
        ):
            value = state.get(key)
            setattr(self, attr, None if value is None else float(value))
        for attr, key in (
            ("_pending_successes", "pending_successes"),
            ("_pending_failures", "pending_failures"),
            ("_pending_timeouts", "pending_timeouts"),
            ("_completed_successes", "completed_successes"),
            ("_completed_failures", "completed_failures"),
            ("_completed_timeouts", "completed_timeouts"),
            ("_completed_success_steps", "completed_success_steps"),
        ):
            tensor = getattr(self, attr)
            tensor.fill_(int(state.get(key, 0) or 0))

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)

        self._record_reset_outcomes(env_ids)
        num_reset = self._sampled_magnitude[env_ids].numel()
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
