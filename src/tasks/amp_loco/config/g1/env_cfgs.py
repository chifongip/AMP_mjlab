"""Unitree G1 AMP Locomotion environment configurations."""

import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

import src.tasks.amp_loco.mdp as amp_mdp
from src.assets.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from src.assets.robots.unitree_g1.g1_constants import HOME_KEYFRAME
from src.tasks.amp_loco.amp_env_cfg import make_amp_env_cfg

def g1_amp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_amp_env_cfg()

  # Keep CCD high enough for stability but avoid Warp OOM from excessive EPA buffers.
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )
  body_names = ("pelvis",
                "left_hip_roll_link",
                "left_knee_link",
                "left_ankle_roll_link",
                "right_hip_roll_link",
                "right_knee_link",
                "right_ankle_roll_link",
                "left_shoulder_roll_link",
                "left_elbow_link",
                "left_wrist_yaw_link",
                "right_shoulder_roll_link",
                "right_elbow_link",
                "right_wrist_yaw_link",)
  anchor_name = "torso_link"
  root_name = "pelvis"

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Configure motion reset to sample from the entire motion with a delay.
  cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.4
  cfg.events["init_motion_loader"].params["max_delay_steps"] = 250

  # Set motion data path for startup loader and reset.
  _motion_base = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "assets", "motions", "g1", "amp"
  )
  _motion_dir = os.path.abspath(os.path.join(_motion_base, "WalkandRun"))
  _recovery_dir = os.path.abspath(os.path.join(_motion_base, "Recovery"))

  cfg.events["init_motion_loader"].params["motion_dir"] = _motion_dir
  cfg.events["init_motion_loader"].params["recovery_dir"] = _recovery_dir
  cfg.events["reset_from_motion"].params["motion_dir"] = _motion_dir

  cfg.rewards["track_anchor_linear_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.rewards["track_anchor_angular_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  cfg.rewards["body_ang_vel_xy_l2"].params["body_cfg"].body_names = (root_name,)

  cfg.observations["critic"].terms["body_pos_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["critic"].terms["body_pos_b"].params["body_cfg"].body_names = body_names
 
  cfg.observations["critic"].terms["body_ori_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["critic"].terms["body_ori_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_pos_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_pos_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_ori_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_ori_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_lin_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_lin_vel_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_ang_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_ang_vel_b"].params["body_cfg"].body_names = body_names

  

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 1.0

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def g1_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = g1_amp_rough_env_cfg(play=play)

  cfg.sim.njmax = 640
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 3.0)
    twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-3.14 / 2, 3.14 / 2)

  return cfg


def g1_amp_recovery_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain fall-recovery configuration."""
  cfg = g1_amp_flat_env_cfg(play=play)

  # Mix recovery resets with quiet standing resets using a fixed startup mask.
  cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 1.0 if play else 0.9
  cfg.events["init_motion_loader"].params["max_delay_steps"] = 250
  cfg.events["reset_from_motion"].params["home_keyframe"] = HOME_KEYFRAME

  # Recovery ends in quiet standing rather than commanded locomotion.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_standing_envs = 1.0
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.heading_command = False
  twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = None

  # Velocity and terrain curricula are not part of recovery-only training.
  cfg.curriculum = {}

  cfg.rewards["track_root_height"].params.update(
    {
      "target_height": HOME_KEYFRAME.pos[2],
      "reward_outside_delay": True,
    }
  )

  # Once upright, settle into the HOME pose represented in the AMP references.
  cfg.rewards["stand_still"] = RewardTermCfg(
    func=amp_mdp.StandStill,
    weight=-1.0,
    params={
      "target_joint_pos": HOME_KEYFRAME.joint_pos,
      "mask_delay": True,
      "delay_env_rew_ratio": 0.0,
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )
  cfg.rewards["body_orientation_l2"] = RewardTermCfg(
    func=amp_mdp.body_orientation_l2,
    weight=-1.0,
    params={
      "mask_delay": True,
      "delay_env_rew_ratio": 0.0,
      "asset_cfg": SceneEntityCfg("robot"),
      "body_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.metrics["root_height"] = MetricsTermCfg(
    func=amp_mdp.root_height,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.metrics["torso_tilt"] = MetricsTermCfg(
    func=amp_mdp.body_tilt,
    params={
      "body_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.metrics["standing_success"] = MetricsTermCfg(
    func=amp_mdp.standing_success,
    params={
      "minimum_height": 0.7,
      "maximum_tilt": math.radians(20.0),
      "asset_cfg": SceneEntityCfg("robot"),
      "body_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )

  if not play:
    cfg.events["upward_recovery_assistance"] = EventTermCfg(
      func=amp_mdp.UpwardRecoveryAssistance,
      mode="step",
      params={
        "force_range": (0.0, 200.0),
        # 2000 policy iterations at 24 environment steps per iteration.
        "anneal_steps": 2000 * 24,
        "asset_cfg": SceneEntityCfg(
          "robot", body_names=("torso_link",)
        ),
        # Draw an orange +Z arrow in viewers/videos when assistance is active.
        "debug_vis": True,
        # 200 N maps to a 0.4 m arrow, keeping the overlay readable.
        "viz_scale": 0.002,
        "viz_width": 0.02,
      },
    )

  return cfg
