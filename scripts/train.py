"""Script to train RL agent with RSL-RL."""

import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Literal

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

from src.tasks.amp_loco.recovery_curriculum import (
  configure_recovery_curriculum_interval,
  validate_recovery_curriculum_workers,
)


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  motion_file: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  deterministic: bool = False
  """Favor practically reproducible GPU execution over maximum throughput."""
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def _package_version(distribution: str) -> str | None:
  try:
    return metadata.version(distribution)
  except metadata.PackageNotFoundError:
    return None


def _git_value(*args: str) -> str | None:
  try:
    result = subprocess.run(
      ("git", *args),
      check=True,
      capture_output=True,
      text=True,
    )
  except (OSError, subprocess.CalledProcessError):
    return None
  return result.stdout.strip()


def _write_provenance(
  log_dir: Path,
  task_id: str,
  cfg: TrainConfig,
  device: str,
) -> None:
  import torch

  nvidia_driver = None
  driver_path = Path("/proc/driver/nvidia/version")
  if driver_path.exists():
    nvidia_driver = driver_path.read_text(encoding="utf-8").splitlines()[0]

  gpu = None
  if device.startswith("cuda") and torch.cuda.is_available():
    gpu_index = torch.device(device).index or 0
    properties = torch.cuda.get_device_properties(gpu_index)
    gpu = {
      "name": properties.name,
      "compute_capability": f"{properties.major}.{properties.minor}",
      "total_memory_bytes": properties.total_memory,
    }

  status = _git_value("status", "--short")
  provenance = {
    "task_id": task_id,
    "command": sys.argv,
    "working_directory": str(Path.cwd()),
    "seed": cfg.agent.seed,
    "device": device,
    "deterministic_requested": cfg.deterministic,
    "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
    "cudnn_deterministic": torch.backends.cudnn.deterministic,
    "cudnn_benchmark": torch.backends.cudnn.benchmark,
    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    "python": platform.python_version(),
    "platform": platform.platform(),
    "git": {
      "commit": _git_value("rev-parse", "HEAD"),
      "dirty": bool(status),
      "status": status,
    },
    "gpu": gpu,
    "nvidia_driver": nvidia_driver,
    "torch": {
      "version": torch.__version__,
      "cuda": torch.version.cuda,
      "cudnn": torch.backends.cudnn.version(),
    },
    "packages": {
      name: _package_version(name)
      for name in (
        "mjlab",
        "mujoco",
        "mujoco-warp",
        "warp-lang",
        "rsl-rl-lib",
        "wbc_mjlab",
      )
    },
  }
  params_dir = log_dir / "params"
  params_dir.mkdir(parents=True, exist_ok=True)
  (params_dir / "provenance.json").write_text(
    json.dumps(provenance, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends(
    allow_tf32=not cfg.deterministic,
    deterministic=cfg.deterministic,
  )
  if cfg.deterministic:
    import torch

    # warn_only keeps the practical GPU mode usable when MuJoCo Warp reaches
    # an operation for which PyTorch has no deterministic implementation.
    torch.use_deterministic_algorithms(True, warn_only=True)

  cfg.agent.seed = seed
  cfg.env.seed = seed
  configure_recovery_curriculum_interval(
    cfg.env.events,
    cfg.agent.num_steps_per_env,
  )
  if cfg.deterministic:
    for sensor_cfg in cfg.env.scene.sensors:
      if getattr(sensor_cfg, "reduce", None) == "none":
        # Unsorted contact slots are explicitly non-deterministic. Choosing
        # the strongest contact gives stable semantics for single-slot sensors.
        sensor_cfg.reduce = "maxforce"

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    if not cfg.motion_file:
      raise ValueError("For tracking tasks, --motion-file must be set ...")
    motion_path = Path(cfg.motion_file).expanduser().resolve()
    if not motion_path.exists():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(motion_path)
    print(f"[INFO] Using motion file: {motion_cmd.motion_file}")

    # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    if cfg.deterministic:
      print(
        "[INFO] Practical determinism enabled; MuJoCo Warp does not guarantee "
        "bitwise-identical simulation."
      )

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(  # 写一个自己的包装器，用于motion tracking
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions) # 因为我要接上自己的rsl_rl，所以要重新写一个包装器

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner   # 

  runner_kwargs = {}
  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)
    _write_provenance(log_dir, task_id, cfg, device)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  if args.deterministic:
    # This must be set before the first CUDA/cuBLAS operation in each worker.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)
  validate_recovery_curriculum_workers(args.env.events, num_gpus)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY
      + ("MUJOCO*", "CUBLAS_WORKSPACE_CONFIG"),
    ).run(run_train, task_id, args, log_dir)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
