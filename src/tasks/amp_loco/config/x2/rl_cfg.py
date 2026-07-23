"""RL configuration for the Agibot X2 AMP recovery task."""

import os

from src.tasks.amp_loco.config.g1.rl_cfg import (
  RslRlAmpRunnerCfg,
  g1_amp_recovery_ppo_runner_cfg,
)


def x2_amp_recovery_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
  """Create the Agibot X2 AMP recovery runner configuration."""
  cfg = g1_amp_recovery_ppo_runner_cfg()
  cfg.experiment_name = "x2_amp_recovery"
  cfg.amp_motion_files = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "assets", "motions", "x2", "amp", "Recovery",
  ))
  cfg.min_normalized_std = [0.05] * 29
  cfg.amp_body_names = (
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
  return cfg
