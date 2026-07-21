"""Generate a stationary AMP reference clip from the G1 HOME keyframe."""

from pathlib import Path
import re

import mujoco
import numpy as np
import tyro

from src.assets.robots.unitree_g1.g1_constants import HOME_KEYFRAME, get_spec


def main(
  output_file: str = (
    "src/assets/motions/g1/amp/Recovery/Stand/home_stand.npz"
  ),
  fps: float = 50.0,
  duration_s: float = 2.0,
) -> None:
  """Generate a static HOME-pose clip with MuJoCo forward kinematics."""
  if fps <= 0.0:
    raise ValueError("fps must be positive.")
  if duration_s <= 0.0:
    raise ValueError("duration_s must be positive.")

  model = get_spec().compile()
  data = mujoco.MjData(model)

  floating_joint = model.joint("floating_base_joint")
  root_qpos_adr = int(floating_joint.qposadr[0])
  data.qpos[root_qpos_adr : root_qpos_adr + 3] = HOME_KEYFRAME.pos
  data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = HOME_KEYFRAME.rot

  joint_names: list[str] = []
  joint_pos: list[float] = []
  for joint_id in range(model.njnt):
    joint = model.joint(joint_id)
    if joint.name == "floating_base_joint":
      continue

    value = 0.0
    for pattern, pattern_value in HOME_KEYFRAME.joint_pos.items():
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

  output_path = Path(output_file)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    output_path,
    fps=np.asarray([fps], dtype=np.float64),
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
