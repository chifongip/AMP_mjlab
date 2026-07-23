"""Generate a stationary AMP reference clip from a robot HOME keyframe."""

from pathlib import Path
import re
from typing import Literal

import mujoco
import numpy as np
import tyro

from mjlab.entity import EntityCfg

from src.assets.robots.agibot_x2.x2_constants import (
  HOME_KEYFRAME as X2_HOME_KEYFRAME,
)
from src.assets.robots.agibot_x2.x2_constants import get_spec as get_x2_spec
from src.assets.robots.unitree_g1.g1_23dof_constants import (
  HOME_KEYFRAME as G1_23DOF_HOME_KEYFRAME,
)
from src.assets.robots.unitree_g1.g1_23dof_constants import (
  get_spec as get_g1_23dof_spec,
)
from src.assets.robots.unitree_g1.g1_constants import (
  HOME_KEYFRAME as G1_HOME_KEYFRAME,
)
from src.assets.robots.unitree_g1.g1_constants import get_spec as get_g1_spec


RobotName = Literal["g1", "g1_23dof", "x2"]


def get_robot_definition(
  robot: RobotName,
) -> tuple[mujoco.MjSpec, EntityCfg.InitialStateCfg]:
  """Return the MJCF specification and HOME keyframe for a robot."""
  if robot == "g1":
    return get_g1_spec(), G1_HOME_KEYFRAME
  if robot == "g1_23dof":
    return get_g1_23dof_spec(), G1_23DOF_HOME_KEYFRAME
  return get_x2_spec(), X2_HOME_KEYFRAME


def main(
  robot: RobotName = "g1",
  output_file: str | None = None,
  fps: float = 50.0,
  duration_s: float = 2.0,
) -> None:
  """Generate a static HOME-pose clip with MuJoCo forward kinematics.

  Args:
    robot: Robot whose MJCF and HOME keyframe define the motion.
    output_file: Destination NPZ. Defaults to the robot's AMP recovery directory.
    fps: Motion frame rate.
    duration_s: Motion duration in seconds.
  """
  if fps <= 0.0:
    raise ValueError("fps must be positive.")
  if duration_s <= 0.0:
    raise ValueError("duration_s must be positive.")

  spec, home_keyframe = get_robot_definition(robot)
  model = spec.compile()
  data = mujoco.MjData(model)

  floating_joint = model.joint("floating_base_joint")
  root_qpos_adr = int(floating_joint.qposadr[0])
  data.qpos[root_qpos_adr : root_qpos_adr + 3] = home_keyframe.pos
  data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = home_keyframe.rot

  joint_names: list[str] = []
  joint_pos: list[float] = []
  for joint_id in range(model.njnt):
    joint = model.joint(joint_id)
    if joint.name == "floating_base_joint":
      continue

    value = 0.0
    for pattern, pattern_value in home_keyframe.joint_pos.items():
      if re.fullmatch(pattern, joint.name):
        value = pattern_value
        break
    data.qpos[int(joint.qposadr[0])] = value
    joint_names.append(joint.name)
    joint_pos.append(value)

  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)

  num_frames = max(2, int(round(fps * duration_s)))
  num_bodies = model.nbody - 1  # Exclude the MuJoCo world body.
  body_slice = slice(1, model.nbody)

  joint_pos_frame = np.asarray(joint_pos, dtype=np.float32)
  joint_vel_frame = np.zeros_like(joint_pos_frame)
  body_pos_frame = np.asarray(data.xpos[body_slice], dtype=np.float32)
  body_quat_frame = np.asarray(data.xquat[body_slice], dtype=np.float32)
  body_vel_frame = np.zeros((num_bodies, 3), dtype=np.float32)

  if output_file is None:
    output_file = (
      f"src/assets/motions/{robot}/amp/Recovery/Stand/home_stand.npz"
    )
  output_path = Path(output_file)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    output_path,
    fps=np.asarray([fps], dtype=np.float64),
    joint_names=np.asarray(joint_names, dtype=str),
    joint_pos=np.repeat(joint_pos_frame[None, :], num_frames, axis=0),
    joint_vel=np.repeat(joint_vel_frame[None, :], num_frames, axis=0),
    body_pos_w=np.repeat(body_pos_frame[None, :, :], num_frames, axis=0),
    body_quat_w=np.repeat(body_quat_frame[None, :, :], num_frames, axis=0),
    body_lin_vel_w=np.repeat(body_vel_frame[None, :, :], num_frames, axis=0),
    body_ang_vel_w=np.repeat(body_vel_frame[None, :, :], num_frames, axis=0),
  )
  print(
    f"Saved {num_frames} HOME frames for {len(joint_names)} joints to "
    f"{output_path}"
  )


if __name__ == "__main__":
  tyro.cli(main)
