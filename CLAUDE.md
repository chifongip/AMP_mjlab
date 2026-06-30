# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AMP_mjlab is a reinforcement learning project for Unitree G1 humanoid locomotion and fall-recovery, built on `mjlab` (MuJoCo-based sim) + custom `rsl_rl`. A single unified policy learns walk/run locomotion and recovery (fall-and-get-up) together, regularized by an AMP discriminator. The pipeline supports direct ONNX export for deployment.

## Key Directories

- `src/tasks/amp_loco/` — AMP locomotion/recovery task implementation
  - `amp_env_cfg.py` — Factory `make_amp_env_cfg()` that builds the base env config (observations, rewards, commands, events, terminations)
  - `ampmotion_loader.py` — NPZ motion data loader
  - `mdp/` — Custom reward/observation/event/termination terms (rewards.py, events.py, terrain.py, etc.)
  - `config/g1/` — G1-specific env and RL config registration (`env_cfgs.py`, `rl_cfg.py`)
  - `rl/runner.py` — `AMPOnPolicyRunner` with ONNX export (+ auto-save during training)
- `src/tasks/velocity/` — Base velocity task (parent of AMP task, used for reference)
- `src/assets/robots/unitree_g1/` — G1 robot constants, actuators, dof definitions
- `src/assets/motions/g1/amp/` — Motion data: `WalkandRun/` and `Recovery/` (NPZ clips)
- `rsl_rl/` — External dependency: [chifongip/rsl_rl](https://github.com/chifongip/rsl_rl) (standalone fork with AMP support — AMPPPO algorithm, AMP loader, discriminator). Installed separately, not vendored.
- `scripts/` — CLI entry points (`train.py`, `play.py`, `csv_to_npz.py`, `list_envs.py`)
- `mjlab_patch/` — Optional patch for mjlab's `observation_manager.py` (adds `time`-ordered history flattening)

## Architecture

**Training pipeline**: `scripts/train.py` → registers task → loads env/RL configs → creates `ManagerBasedRlEnv` → wraps with `RslRlVecEnvWrapper` → instantiates runner (`AMPOnPolicyRunner` for AMP tasks) → calls `runner.learn()`. The runner initializes `AMPPPO` algorithm with an `AMP discriminator` and `AMPLoader`, runs PPO rollouts, and applies AMP discriminator reward.

**Key config flow**: `config/g1/env_cfgs.py` calls `make_amp_env_cfg()` (base factory in `amp_env_cfg.py`), then overrides per-robot settings (terrain, sensors, reward params, motion dirs). `config/g1/rl_cfg.py` defines `RslRlAmpRunnerCfg` with AMP-specific fields (`amp_reward_coef`, `amp_motion_files`, `amp_discr_hidden_dims`, etc.).

**Delayed reset mechanism**: `mdp/events.py` implements `MotionResetManager` (singleton) that loads Walk/Run and Recovery motion frames, then on reset splits envs into "delayed" (recovery) and "normal" (walk/run). `mdp/terminations.py` provides `DelayedTerminationManager` that wraps the termination manager with a delay counter.

**Reward system** (`mdp/rewards.py`): Key rewards include velocity tracking, root height tracking, body angular velocity penalty, foot slip, joint acceleration/position limits, action rate, self-collision cost, and is_terminated penalty. Delay-aware reward scaling via `_apply_delay_env_reward_scaling`.

**ONNX export**: Both training (auto-save) and play support ONNX export. `AMPOnPolicyRunner._export_policy_to_onnx()` wraps actor_critic + obs_normalizer in a Module and exports with opset 18. The exported model is self-contained (external data inlined).

## Commands

```bash
# Install
conda activate mjlab
pip install git+https://github.com/chifongip/rsl_rl.git@main  # rsl_rl with AMP support
cd AMP_mjlab
python -m pip install -e .

# List available tasks
python scripts/list_envs.py --keyword AMP

# Train (AMP-Flat: flat terrain)
python scripts/train.py Unitree-G1-AMP-Flat

# Train (AMP-Rough: rough terrain)
python scripts/train.py Unitree-G1-AMP-Rough

# Evaluate / play with a trained checkpoint
python scripts/play.py Unitree-G1-AMP-Rough \
  --checkpoint-file logs/rsl_rl/g1_amp_locomotion/<run_dir>/model_<iter>.pt

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py Unitree-G1-AMP-Flat

# Motion data conversion
python scripts/csv_to_npz.py --help
```

Logs go to `logs/rsl_rl/g1_amp_locomotion/<timestamp>/`. ONNX exports are saved to `logs/rsl_rl/g1_amp_locomotion/<timestamp>/policy.onnx`.

## Development Notes

- The `AMPOnPolicyRunner` (in `src/tasks/amp_loco/rl/runner.py`) extends `AmpOnPolicyRunner` from the standalone rsl_rl fork and adds a config adapter (`_adapt_amp_config()`) plus auto-ONNX export on every save.
- The AMPPPO algorithm (in the rsl_rl fork) combines standard PPO returns with AMP discriminator rewards.
- Observation history uses term-major ordering (4-step history) — `history_ordering="time"` was removed for mjlab 1.5.0 compatibility.
- Training typically shows a sudden transition around ~20k iterations when the policy learns recovery behavior — this is expected behavior, not a bug.
- Deployment code is in a separate repo: `ccrpRepo/wbc_fsm` (MJAmp State).
