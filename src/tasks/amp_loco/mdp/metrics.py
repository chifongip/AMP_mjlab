from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
RECOVERY_MINIMUM_HEIGHT = 0.7
RECOVERY_MAXIMUM_TILT = math.radians(20.0)
RECOVERY_STABLE_STEPS = 25


def root_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return root height relative to each environment origin."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.body_link_pos_w[:, 0, 2] - env.scene.env_origins[:, 2]


def body_tilt(
  env: ManagerBasedRlEnv,
  body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """Return body tilt from upright in radians."""
  asset: Entity = env.scene[body_cfg.name]
  body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids[0]]
  return body_tilt_from_state(body_quat_w, asset.data.gravity_vec_w)


def body_tilt_from_state(
  body_quat_w: torch.Tensor,
  gravity_vec_w: torch.Tensor,
) -> torch.Tensor:
  """Return body tilt from already-resolved orientation tensors."""
  projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_vec_w)
  return torch.acos(
    torch.clamp(-projected_gravity_b[:, 2], min=-1.0, max=1.0)
  )


def standing_success(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  maximum_tilt: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """Return one for environments satisfying standing height and tilt targets."""
  return standing_mask(
    env,
    minimum_height=minimum_height,
    maximum_tilt=maximum_tilt,
    asset_cfg=asset_cfg,
    body_cfg=body_cfg,
  ).float()


def standing_mask(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  maximum_tilt: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """Return a boolean mask for environments satisfying standing targets."""
  height = root_height(env, asset_cfg)
  asset: Entity = env.scene[body_cfg.name]
  body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids[0]]
  return standing_mask_from_state(
    height,
    body_quat_w,
    asset.data.gravity_vec_w,
    minimum_height,
    maximum_tilt,
  )


def standing_mask_from_state(
  root_height_w: torch.Tensor,
  body_quat_w: torch.Tensor,
  gravity_vec_w: torch.Tensor,
  minimum_height: float,
  maximum_tilt: float | torch.Tensor,
) -> torch.Tensor:
  """Return standing state from already-resolved body tensors."""
  tilt = body_tilt_from_state(body_quat_w, gravity_vec_w)
  return (root_height_w >= minimum_height) & (tilt <= maximum_tilt)


def _recovery_cohort_mean(
  env: ManagerBasedRlEnv,
  values: torch.Tensor,
) -> torch.Tensor:
  """Return the recovery cohort mean expanded to the manager's required shape."""
  recovery_mask = getattr(env.termination_manager, "_delay_env_mask", None)
  if not isinstance(recovery_mask, torch.Tensor):
    return torch.zeros(env.num_envs, device=env.device)
  mask = recovery_mask.float()
  cohort_mean = torch.sum(values.float() * mask) / torch.clamp(
    torch.sum(mask), min=1.0
  )
  return cohort_mean.expand(env.num_envs)


def recovery_standing_success(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  maximum_tilt: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """Current standing occupancy among recovery-reset environments."""
  values = standing_success(
    env,
    minimum_height=minimum_height,
    maximum_tilt=maximum_tilt,
    asset_cfg=asset_cfg,
    body_cfg=body_cfg,
  )
  return _recovery_cohort_mean(env, values)


def recovery_root_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Mean root height among recovery-reset environments only."""
  return _recovery_cohort_mean(env, root_height(env, asset_cfg))


def recovery_body_tilt(
  env: ManagerBasedRlEnv,
  body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """Mean torso tilt among recovery-reset environments only."""
  return _recovery_cohort_mean(env, body_tilt(env, body_cfg))


def recovery_failure_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Current delayed-failure occupancy among recovery environments."""
  counters = getattr(env.termination_manager, "_delay_counters", None)
  if not isinstance(counters, torch.Tensor):
    return torch.zeros(env.num_envs, device=env.device)
  return _recovery_cohort_mean(env, counters > 0)


def recovery_timeout_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Current per-step timeout incidence among recovery environments."""
  time_outs = env.termination_manager.time_outs
  return _recovery_cohort_mean(env, time_outs)


def mean_delay_steps(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Mean number of delay steps for environments in delayed termination.

  This metric is useful for monitoring the delay duration when using delayed termination.

  Returns:
    Per-environment scalar. Shape: ``(B,)``.
  """
  tm = env.termination_manager
  delay_counters = getattr(tm, "_delay_counters", None)
  delay_env_mask = getattr(tm, "_delay_env_mask", None)
  
  if delay_env_mask is not None and delay_counters is not None:
    total_delay_steps = torch.sum(delay_counters.float())
    total_delay_envs = torch.sum(delay_env_mask.float())
    mean_delay = total_delay_steps / (total_delay_envs + 1e-8)  # Avoid division by zero.
    return mean_delay.expand(env.num_envs)  # Return same mean for all envs for easier logging.
  else:
    return torch.zeros(env.num_envs, device=env.device)
