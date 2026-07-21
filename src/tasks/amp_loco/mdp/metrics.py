from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


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
  projected_gravity_b = quat_apply_inverse(body_quat_w, asset.data.gravity_vec_w)
  return torch.acos(torch.clamp(-projected_gravity_b[:, 2], min=-1.0, max=1.0))


def standing_success(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  maximum_tilt: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """Return one for environments satisfying standing height and tilt targets."""
  height = root_height(env, asset_cfg)
  tilt = body_tilt(env, body_cfg)
  return ((height >= minimum_height) & (tilt <= maximum_tilt)).float()


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
