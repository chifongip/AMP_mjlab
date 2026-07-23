"""Agibot X2 AMP recovery environment configuration."""

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

from src.assets.robots import get_agibot_x2_robot_cfg
from src.assets.robots.agibot_x2.x2_constants import HOME_KEYFRAME
from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_recovery_flat_env_cfg


_RECOVERY_MINIMUM_HEIGHT = 0.58

_AMP_BODY_NAMES = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_roll_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_roll_link",
)


def x2_amp_recovery_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the Agibot X2 flat-terrain recovery configuration."""
  cfg = g1_amp_recovery_flat_env_cfg(play=play)

  robot_cfg, action_scale = get_agibot_x2_robot_cfg(preset="agibot_stiff")
  cfg.scene.entities = {"robot": robot_cfg}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = action_scale

  motion_base = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "assets", "motions", "x2", "amp", "Recovery",
  ))
  stand_motion_dir = os.path.join(motion_base, "Stand")
  cfg.events["init_motion_loader"].params["motion_dir"] = stand_motion_dir
  cfg.events["init_motion_loader"].params["recovery_dir"] = motion_base
  cfg.events["reset_from_motion"].params["motion_dir"] = stand_motion_dir
  cfg.events["reset_from_motion"].params["home_keyframe"] = HOME_KEYFRAME

  cfg.terminations["bad_base_height"].params["minimum_height"] = (
    _RECOVERY_MINIMUM_HEIGHT
  )
  cfg.rewards["track_root_height"].params["target_height"] = HOME_KEYFRAME.pos[2]
  cfg.rewards["stand_still"].params["target_joint_pos"] = HOME_KEYFRAME.joint_pos

  for group_name in ("critic", "amp"):
    group = cfg.observations[group_name]
    for term_name in ("body_pos_b", "body_ori_b"):
      group.terms[term_name].params["body_cfg"].body_names = _AMP_BODY_NAMES
  for term_name in ("body_lin_vel_b", "body_ang_vel_b"):
    cfg.observations["amp"].terms[term_name].params["body_cfg"].body_names = (
      _AMP_BODY_NAMES
    )

  cfg.metrics["standing_success"].params["minimum_height"] = (
    _RECOVERY_MINIMUM_HEIGHT
  )
  cfg.metrics["recovery_standing_occupancy"].params["minimum_height"] = (
    _RECOVERY_MINIMUM_HEIGHT
  )

  assistance = cfg.events.get("upward_recovery_assistance")
  if assistance is not None:
    assistance.params["minimum_height"] = _RECOVERY_MINIMUM_HEIGHT

  return cfg
