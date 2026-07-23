"""Unitree G1 23-DOF AMP recovery environment configuration."""

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

from src.assets.robots import (
  G1_23DOF_ACTION_SCALE,
  get_g1_23dof_robot_cfg,
)
from src.assets.robots.unitree_g1.g1_23dof_constants import HOME_KEYFRAME
from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_recovery_flat_env_cfg


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
  "left_wrist_roll_rubber_hand",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_roll_rubber_hand",
)


def g1_23dof_amp_recovery_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the Unitree G1 23-DOF flat-terrain recovery configuration."""
  cfg = g1_amp_recovery_flat_env_cfg(play=play)

  cfg.scene.entities = {"robot": get_g1_23dof_robot_cfg()}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_23DOF_ACTION_SCALE

  motion_base = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "assets", "motions", "g1_23dof", "amp", "Recovery",
  ))
  stand_motion_dir = os.path.join(motion_base, "Stand")
  cfg.events["init_motion_loader"].params["motion_dir"] = stand_motion_dir
  cfg.events["init_motion_loader"].params["recovery_dir"] = motion_base
  cfg.events["reset_from_motion"].params["motion_dir"] = stand_motion_dir
  cfg.events["reset_from_motion"].params["home_keyframe"] = HOME_KEYFRAME

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

  return cfg
