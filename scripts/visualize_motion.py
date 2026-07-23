#!/usr/bin/env python3
"""Visualize NPZ motion data in MuJoCo.

Replays a motion clip from an NPZ file on the matching robot model in
an interactive MuJoCo passive viewer window.  Supports all three robot
variants used by the AMP pipeline (g1 / g1_23dof / x2).

Usage:
  # Walk forward motion (auto-detect g1 from joint count)
  python scripts/visualize_motion.py \
    --file src/assets/motions/g1/amp/WalkandRun/walk_forward_loop_002__A022.npz

  # Recovery motion at half real-time speed
  python scripts/visualize_motion.py \
    --file src/assets/motions/g1/amp/Recovery/fallAndGetUp1_subject1.npz \
    --realtime --realtime-scale 0.5

  # 23-DOF variant (auto-detected)
  python scripts/visualize_motion.py \
    --file src/assets/motions/g1_23dof/amp/Recovery/fallAndGetUp1_subject1_g1_23dof.npz

  # X2 robot (must be explicit – same joint count as g1)
  python scripts/visualize_motion.py \
    --file src/assets/motions/x2/amp/Recovery/fallAndGetUp1_subject1_agibot_x2.npz \
    --robot x2

  # Play once and exit (no loop)
  python scripts/visualize_motion.py \
    --file src/assets/motions/g1/amp/WalkandRun/jog_forward_loop_003__A021.npz \
    --no-loop
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import mujoco
import mujoco.viewer as mj_viewer
import numpy as np
import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

from src.assets.robots import get_agibot_x2_robot_cfg, get_g1_23dof_robot_cfg

RobotName = Literal["auto", "g1", "g1_23dof", "x2"]


# ---------------------------------------------------------------------------
# Robot detection
# ---------------------------------------------------------------------------


def _detect_robot(npz_path: str, robot_override: RobotName) -> str:
  """Return the robot name (`g1` / `g1_23dof` / `x2`) for *npz_path*."""
  if robot_override != "auto":
    return robot_override

  # Peek at joint count – np.load is fast enough for a metadata peek.
  with np.load(npz_path) as data:
    joint_count: int = data["joint_pos"].shape[1]

  if joint_count == 23:
    return "g1_23dof"

  # 29 joints → g1 or x2.  Disambiguate via path.
  path_lower = str(npz_path).lower()
  if "x2" in path_lower or "agibot" in path_lower:
    return "x2"
  return "g1"


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def _make_scene(robot_name: str, device: str) -> Scene:
  """Build a single-robot replay Scene for *robot_name*."""
  if robot_name == "g1":
    return Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)

  if robot_name == "g1_23dof":
    robot_cfg = get_g1_23dof_robot_cfg()
  else:  # x2
    robot_cfg, _ = get_agibot_x2_robot_cfg()

  return Scene(SceneCfg(entities={"robot": robot_cfg}), device=device)


def _configure_camera(viewer, distance: float, elevation: float, azimuth: float):
  """Set initial camera parameters on the passive viewer."""
  viewer.cam.distance = distance
  viewer.cam.elevation = elevation
  viewer.cam.azimuth = azimuth


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _run_viewer(
  sim: Simulation,
  scene: Scene,
  npz_data: dict[str, np.ndarray],
  fps: float,
  *,
  realtime: bool,
  realtime_scale: float,
  loop: bool,
  camera_distance: float,
  camera_elevation: float,
  camera_azimuth: float,
) -> None:
  """Replay *npz_data* frame-by-frame in a passive MuJoCo viewer."""
  robot: Entity = scene["robot"]
  device = sim.device

  total_frames: int = npz_data["joint_pos"].shape[0]
  sim_dt = 1.0 / fps

  print(f"Motion: {total_frames} frames @ {fps} fps  "
        f"({total_frames / fps:.1f} s)")
  print(f"Robot:  {len(robot.joint_names)} joints")
  print("Close the MuJoCo window to exit.\n")

  with mj_viewer.launch_passive(sim.mj_model, sim.mj_data) as viewer:
    _configure_camera(viewer, camera_distance, camera_elevation, camera_azimuth)

    frame_idx = 0
    wall_start = time.perf_counter()

    while viewer.is_running():
      #
      # Write root state (body index 0)
      #
      root_pos = torch.from_numpy(
        npz_data["body_pos_w"][frame_idx, 0:1, :]
      ).float().to(device)
      root_quat = torch.from_numpy(
        npz_data["body_quat_w"][frame_idx, 0:1, :]
      ).float().to(device)
      root_lin_vel = torch.from_numpy(
        npz_data["body_lin_vel_w"][frame_idx, 0:1, :]
      ).float().to(device)
      root_ang_vel = torch.from_numpy(
        npz_data["body_ang_vel_w"][frame_idx, 0:1, :]
      ).float().to(device)

      root_states = robot.data.default_root_state.clone()
      root_states[:, 0:3] = root_pos
      root_states[:, :2] += scene.env_origins[:, :2]
      root_states[:, 3:7] = root_quat
      root_states[:, 7:10] = root_lin_vel
      root_states[:, 10:13] = root_ang_vel
      robot.write_root_state_to_sim(root_states)

      #
      # Write joint state
      #
      joint_pos = torch.from_numpy(
        npz_data["joint_pos"][frame_idx]
      ).float().to(device).unsqueeze(0)
      joint_vel = torch.from_numpy(
        npz_data["joint_vel"][frame_idx]
      ).float().to(device).unsqueeze(0)
      robot.write_joint_state_to_sim(joint_pos, joint_vel)

      #
      # Step physics + update scene
      #
      sim.forward()
      scene.update(sim.mj_model.opt.timestep)

      #
      # Sync viewer (copy batch data → mj_data, call mj_forward, draw)
      #
      if sim.mj_model.nq > 0:
        sim.mj_data.qpos[:] = sim.data.qpos[0].cpu().numpy()
        sim.mj_data.qvel[:] = sim.data.qvel[0].cpu().numpy()
      if sim.mj_model.nmocap > 0:
        sim.mj_data.mocap_pos[:] = sim.data.mocap_pos[0].cpu().numpy()
        sim.mj_data.mocap_quat[:] = sim.data.mocap_quat[0].cpu().numpy()
      sim.mj_data.xfrc_applied[:] = sim.data.xfrc_applied[0].cpu().numpy()
      mujoco.mj_forward(sim.mj_model, sim.mj_data)
      viewer.sync()

      #
      # Frame advance + real-time pacing
      #
      frame_idx += 1
      if frame_idx >= total_frames:
        if loop:
          frame_idx = 0
          wall_start = time.perf_counter()
        else:
          print("Reached end of motion (loop disabled).  Close window to exit.")
          # Keep the viewer alive so the user can inspect the last frame,
          # but stop advancing.
          while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
          break

      if realtime:
        sim_elapsed = frame_idx * sim_dt
        target_wall = sim_elapsed / max(realtime_scale, 1e-6)
        now = time.perf_counter() - wall_start
        sleep_s = target_wall - now
        if sleep_s > 0:
          time.sleep(sleep_s)

      # Print progress every 2 simulated seconds.
      if frame_idx % max(1, int(fps * 2)) == 0:
        t = frame_idx * sim_dt
        print(f"  frame {frame_idx:5d} / {total_frames}  "
              f"t = {t:6.1f} s", end="\r")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(
  file: str,
  robot: RobotName = "auto",
  realtime: bool = True,
  realtime_scale: float = 1.0,
  no_loop: bool = False,
  device: str = "cuda:0",
  camera_distance: float = 3.0,
  camera_elevation: float = -10.0,
  camera_azimuth: float = 45.0,
):
  """Visualize an NPZ motion clip on the matching robot in MuJoCo.

  Args:
      file: Path to the NPZ motion file.
      robot: Robot model override (`auto` detects from joint count + path).
      realtime: Pace playback to wall-clock time.
      realtime_scale: Speed multiplier when `--realtime` is set
          (1.0 = real-time, 0.5 = half speed, 2.0 = double speed).
      no_loop: Stop at the last frame instead of looping.
      device: Torch device for simulation tensors.
      camera_distance: Initial camera distance from the robot.
      camera_elevation: Initial camera elevation (degrees).
      camera_azimuth: Initial camera azimuth (degrees).
  """
  npz_path = Path(file).expanduser()
  if not npz_path.is_file():
    raise FileNotFoundError(f"NPZ file not found: {npz_path}")

  robot_name = _detect_robot(str(npz_path), robot)

  # ------------------------------------------------------------------
  # Load NPZ
  # ------------------------------------------------------------------
  print(f"Loading: {npz_path}")
  npz_data = dict(np.load(npz_path))

  required_keys = {
    "fps", "joint_pos", "joint_vel",
    "body_pos_w", "body_quat_w",
    "body_lin_vel_w", "body_ang_vel_w",
  }
  missing = required_keys - set(npz_data.keys())
  if missing:
    raise KeyError(
      f"NPZ file is missing required keys: {sorted(missing)}.  "
      f"Found: {sorted(npz_data.keys())}"
    )

  fps_in_file = float(npz_data["fps"].item())
  joint_count_npz = npz_data["joint_pos"].shape[1]
  print(f"  fps={fps_in_file}, joints={joint_count_npz}, "
        f"frames={npz_data['joint_pos'].shape[0]}")

  # ------------------------------------------------------------------
  # Build simulation
  # ------------------------------------------------------------------
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / fps_in_file

  scene = _make_scene(robot_name, device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  robot_entity: Entity = scene["robot"]
  if len(robot_entity.joint_names) != joint_count_npz:
    raise ValueError(
      f"NPZ joint count ({joint_count_npz}) does not match robot "
      f"'{robot_name}' joint count ({len(robot_entity.joint_names)})."
    )
  print(f"Detected robot: {robot_name}  "
        f"({len(robot_entity.joint_names)} joints)")

  # ------------------------------------------------------------------
  # Replay
  # ------------------------------------------------------------------
  scene.reset()
  _run_viewer(
    sim,
    scene,
    npz_data,
    fps_in_file,
    realtime=realtime,
    realtime_scale=realtime_scale,
    loop=not no_loop,
    camera_distance=camera_distance,
    camera_elevation=camera_elevation,
    camera_azimuth=camera_azimuth,
  )
  print("\nDone.")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
