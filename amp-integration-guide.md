# AMP + mjlab Integration Guide (Quickstart)

> **Audience:** mjlab users who want to bring a new robot or environment into the AMP
> locomotion pipeline. The guide shows you which files to create/modify and how to
> wire everything together.
>
> **Architecture:** The AMP module lives in the rsl_rl v6 repository at
> `/home/ubuntu/rsl_rl` (`rsl_rl.algorithms.AMPPPO`,
> `rsl_rl.runners.AMPOnPolicyRunner`). It uses TensorDict observations,
> separate `MLPModel` actor/critic, and the v6 `RolloutStorage` / `Logger`
> infrastructure.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Step 1 – Create Your Task Package](#step-1--create-your-task-package)
- [Step 2 – Build the Environment Config](#step-2--build-the-environment-config)
- [Step 3 – Add Motion Data](#step-3--add-motion-data)
- [Step 4 – Train](#step-4--train)
- [Appendix A – NPZ Motion Format](#appendix-a--npz-motion-format)
- [Appendix B – ONNX Export](#appendix-b--onnx-export)
- [Appendix C – Rough Terrain (Optional)](#appendix-c--rough-terrain-optional)

---

## Prerequisites

- `mjlab` cloned locally and installed (`pip install -e .`).
- `rsl_rl` v6 with AMP support installed from `/home/ubuntu/rsl_rl`
  (`pip install -e /home/ubuntu/rsl_rl`). Contains `AMPPPO`, `AMPOnPolicyRunner`,
  `Discriminator`, `AMPLoader`, `ReplayBuffer`, `Normalizer`.
- Motion capture data in CSV or NPZ format.
- (Optional) `mjlab_patch/observation_manager.py` applied if you want `time`-ordered
  observation history.

Set your Python path once per shell:

```bash
export MJLAB_PATH="/path/to/mjlab"
export PYTHONPATH="$MJLAB_PATH/src:$PYTHONPATH"
```

### Key API Differences (rsl_rl v6)

| Concept | Old (fork) | New (rsl_rl v6) |
|---------|-----------|-----------------|
| Observations | Plain tensors (`obs`, `critic_obs`, `amp_obs`) | `TensorDict` with keys `"actor"`, `"critic"`, `"amp"` |
| Policy model | Single `ActorCritic` module | Separate `MLPModel` actor + `MLPModel` critic |
| Config format | Dataclass-based (`@configclass`) | Dict-based (`train_cfg["actor"]`, `train_cfg["algorithm"]`) |
| Obs groups | Concatenated flat tensors | Named groups in `TensorDict`, selected via `obs_groups` dict |
| Runner logging | Manual `SummaryWriter` | `Logger` class |
| Algorithm factory | `eval(class_name)` | `AMPPPO.construct_algorithm(obs, env, cfg, device)` |

---

## Step 1 – Create Your Task Package

Create a directory tree that mirrors the existing `src/tasks/` layout. Each task
needs three modules: an env-cfg factory, per-robot configs, and custom MDP terms.

```
src/tasks/your_task/
├── __init__.py                  # docstring only
├── your_task_env_cfg.py         # make_<task>_env_cfg() factory
├── config/
│   ├── __init__.py              # register_mjlab_task() calls
│   └── your_robot/
│       ├── __init__.py          # (optional) re-exports
│       ├── env_cfgs.py          # per-robot env overrides
│       └── rl_cfg.py            # per-robot RL config
└── mdp/
    ├── __init__.py              # re-exports mjlab.envs.mdp + local modules
    ├── observations.py          # custom observation terms (if any)
    ├── rewards.py               # custom reward terms (if any)
    ├── events.py                # motion loader + reset logic
    └── terminations.py          # (optional) custom termination terms
```

> **Tip:** Copy `src/tasks/velocity/` as a starting skeleton if you don't need
> motion-loader integration, or copy `src/tasks/amp_loco/` if you do.

> **Note:** The runner (`AMPOnPolicyRunner`) and algorithm (`AMPPPO`) now live in
> rsl_rl itself — you do **not** need a `rl/runner.py` in your task package.

---

## Step 2 – Build the Environment Config

Your env-cfg factory assembles the full `ManagerBasedRlEnvCfg` dataclass. The
reference implementation is `make_amp_env_cfg()` in
`src/tasks/amp_loco/amp_env_cfg.py`. Here's what it sets up and how to customize.

### 2.1 Reference Factory — `make_amp_env_cfg()`

```python
# src/tasks/amp_loco/amp_env_cfg.py
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg

def make_amp_env_cfg() -> ManagerBasedRlEnvCfg:
    """Assemble a base AMP locomotion env config.

    Returns a ManagerBasedRlEnvCfg with:
      - scene:       SceneCfg (rough terrain, raycast sensor, num_envs=1)
      - observations: three groups — "actor", "critic", "amp"
      - actions:     {"joint_pos": JointPositionActionCfg}
      - commands:    {"twist": UniformVelocityCommandCfg}
      - events:      init_motion_loader, reset_from_motion, push_robot,
                     foot_friction, encoder_bias, base_com
      - rewards:     velocity tracking, root height, joint penalties, etc.
      - terminations: time_out, bad_orientation, bad_base_height
      - metrics:     mean_action_acc, mean_delay_steps
      - sim:         timestep=0.005, decimation=4
      - episode_length_s: 20.0
    """
    cfg = ManagerBasedRlEnvCfg()
    # ... (full implementation in amp_env_cfg.py)
    return cfg
```

**Key point:** The factory takes **no arguments**. It returns a fully populated
config that your per-robot function then overrides.

### 2.2 Per-Robot Config Function

In `config/your_robot/env_cfgs.py`, call the factory and override robot-specific
fields:

```python
# src/tasks/your_task/config/your_robot/env_cfgs.py
from src.tasks.your_task.your_task_env_cfg import make_amp_env_cfg

def your_robot_env_cfg(play: bool = False) -> "ManagerBasedRlEnvCfg":
    """Build env config for YourRobot, overriding AMP defaults."""
    cfg = make_amp_env_cfg()                     # 1. start from factory

    # --- Scene ---
    cfg.scene.robot.spawn.file = "path/to/robot.xml"

    # --- Actions ---
    cfg.actions["joint_pos"].asset_name = "robot"
    cfg.actions["joint_pos"].joint_names = [...]
    cfg.actions["joint_pos"].scale = ...

    # --- Observations: set body names for AMP/Critic groups ---
    # The AMP observation terms (robot_body_pos_b, etc.) need anchor_cfg
    # and body_cfg to know which bodies to track.
    anchor_body = "torso_link"
    tracked_bodies = ["pelvis", "left_foot", "right_foot", ...]

    for group_name in ("critic", "amp"):
        group = cfg.observations[group_name]
        for term_key in ("body_pos_b", "body_ori_b", "body_lin_vel_b", "body_ang_vel_b"):
            if term_key in group:
                group[term_key].anchor_cfg.body_names = [anchor_body]
                group[term_key].body_cfg.body_names = tracked_bodies

    # --- Rewards: set body/sensor names ---
    cfg.rewards["track_anchor_linear_velocity"].anchor_cfg.body_names = [anchor_body]
    cfg.rewards["body_ang_vel_xy_l2"].body_cfg.body_names = ["pelvis"]
    cfg.rewards["foot_slip"].asset_cfg.site_names = ["left_foot_site", "right_foot_site"]
    cfg.rewards["self_collisions"].sensor_name = "self_collision"

    # --- Events: set motion directories (see Step 3) ---
    cfg.events["init_motion_loader"].motion_dir = "src/assets/motions/your_robot/amp/WalkandRun"
    cfg.events["init_motion_loader"].recovery_dir = "src/assets/motions/your_robot/amp/Recovery"
    cfg.events["reset_from_motion"].motion_dir = "src/assets/motions/your_robot/amp/WalkandRun"

    # --- Play mode: fewer envs, no domain randomization ---
    if play:
        cfg.scene.num_envs = 2
        cfg.observations["actor"].enable_corruption = False

    return cfg
```

### 2.3 MDP Modules

In `your_task/mdp/__init__.py`, re-export everything from `mjlab.envs.mdp` and add
your custom terms:

```python
# src/tasks/your_task/mdp/__init__.py
from mjlab.envs.mdp import *  # noqa: F401,F403

from . import events as events
from . import observations as observations
from . import rewards as rewards
from . import terminations as terminations
```

---

## Step 3 – Add Motion Data

AMP learns by comparing policy behavior against expert motion demonstrations.
The motion data flows through three components:

```
NPZ files on disk
    ↓
AMPLoader (loads + preprocesses into flat frame buffer)
    ↓
ReplayBuffer (stores (obs_t, obs_{t+1}) pairs from policy rollouts)
    ↓
Discriminator (distinguishes expert vs policy transitions → reward signal)
```

### 3.1 Directory Structure

Create two directories under your assets folder:

```
src/assets/motions/your_robot/amp/
├── WalkandRun/          # locomotion clips (walk, run, turn)
│   ├── walk_forward.npz
│   ├── run_forward.npz
│   └── ...
└── Recovery/            # fall-recovery clips (optional, for delayed reset)
    ├── getup_front.npz
    └── ...
```

### 3.2 NPZ Format

Each `.npz` file must contain these keys (see [Appendix A](#appendix-a--npz-motion-format)
for full spec):

| Key              | Shape               | Description             |
|------------------|---------------------|-------------------------|
| `fps`            | scalar              | Frames per second       |
| `joint_pos`      | `(T, num_joints)`   | Joint positions         |
| `joint_vel`      | `(T, num_joints)`   | Joint velocities        |
| `body_pos_w`     | `(T, num_bodies, 3)`| Body positions (world)  |
| `body_quat_w`    | `(T, num_bodies, 4)`| Body quaternions (world)|
| `body_lin_vel_w` | `(T, num_bodies, 3)`| Body linear velocities  |
| `body_ang_vel_w` | `(T, num_bodies, 3)`| Body angular velocities |

**Important:** `num_bodies` must match the MJCF model exactly (including root body).
`body_names` in the RL config must list the same bodies in the same order.

### 3.3 How AMPLoader Works

`AMPLoader` (`rsl_rl.utils.motion_loader`) reads all `.npz` files from a directory
and concatenates them into a flat frame buffer. For each frame it computes a
**body-relative observation vector** containing:

- Body positions relative to the anchor body (in anchor frame)
- Body orientations as rotation matrix columns (in anchor frame)
- Body linear velocities relative to the anchor (in anchor frame)
- Body angular velocities relative to the anchor (in anchor frame)

The anchor body is typically the torso or pelvis. The observation dimension is:

```
obs_dim = num_tracked_bodies * (3 pos + 9 ori + 3 lin_vel + 3 ang_vel)
```

For example, with 4 tracked bodies: `obs_dim = 4 * 18 = 72`.

The loader's `feed_forward_generator(num_batches, batch_size)` yields
`(state, next_state)` pairs — consecutive frames from random clips — which the
discriminator uses to learn temporal coherence.

### 3.4 How the Discriminator Works

The `Discriminator` (`rsl_rl.modules.discriminator`) is a simple MLP that takes
concatenated `(state, next_state)` pairs and outputs a scalar logit:

```
Expert transitions  →  target = +1  (discriminator should output high)
Policy transitions  →  target = -1  (discriminator should output low)
```

**Training loss** (per mini-batch):
```python
expert_loss = MSE(discriminator(expert_pair), +1)
policy_loss = MSE(discriminator(policy_pair), -1)
grad_pen    = gradient_penalty(expert_pairs)   # Lipschitz regularization
amp_loss    = 0.5 * (expert_loss + policy_loss) + grad_pen
```

**Reward signal** (during rollout, no gradients):
```python
d = discriminator(policy_state, policy_next_state)
reward = amp_reward_coef * clamp(1 - 0.25 * (d - 1)^2, min=0)
# Optional: blend with task reward
reward = (1 - task_reward_lerp) * reward + task_reward_lerp * task_reward
```

The reward is high when the discriminator thinks the transition looks expert-like,
and low when it looks non-expert. The `amp_reward_coef` scales the magnitude,
and `task_reward_lerp` controls how much task reward (e.g., velocity tracking)
is blended in.

### 3.5 The AMP Observation Group

The `"amp"` observation group in your env config defines what the discriminator
sees. It must contain body-level kinematics that match the motion data:

```python
# In your env config's observations
cfg.observations["amp"] = {
    "body_pos_b": BodyPositionInAnchorFrameCfg(
        anchor_cfg=ObsTermCfg.FuncCfg(body_names=["torso_link"]),
        body_cfg=ObsTermCfg.FuncCfg(body_names=tracked_bodies),
    ),
    "body_ori_b": BodyOrientationInAnchorFrameCfg(
        anchor_cfg=ObsTermCfg.FuncCfg(body_names=["torso_link"]),
        body_cfg=ObsTermCfg.FuncCfg(body_names=tracked_bodies),
    ),
    "body_lin_vel_b": BodyLinearVelocityInAnchorFrameCfg(
        anchor_cfg=ObsTermCfg.FuncCfg(body_names=["torso_link"]),
        body_cfg=ObsTermCfg.FuncCfg(body_names=tracked_bodies),
    ),
    "body_ang_vel_b": BodyAngularVelocityInAnchorFrameCfg(
        anchor_cfg=ObsTermCfg.FuncCfg(body_names=["torso_link"]),
        body_cfg=ObsTermCfg.FuncCfg(body_names=tracked_bodies),
    ),
}
```

**The `amp_body_names` in the RL config must match `tracked_bodies` exactly.**
The `amp_anchor_name` must match the anchor body. These are passed to
`AMPLoader` to compute the same body-relative representation from motion data.

### 3.6 CSV → NPZ Conversion

Use the bundled converter:

```bash
python scripts/csv_to_npz.py \
    --input path/to/your_data.csv \
    --output src/assets/motions/your_robot/amp/WalkandRun/clip.npz \
    --fps 30
```

Run `python scripts/csv_to_npz.py --help` for all options.

### 3.7 Wire Motion Dirs into Config

In your per-robot `env_cfgs.py` (Step 2.2), set:

```python
cfg.events["init_motion_loader"].motion_dir = "src/assets/motions/your_robot/amp/WalkandRun"
cfg.events["init_motion_loader"].recovery_dir = "src/assets/motions/your_robot/amp/Recovery"
cfg.events["reset_from_motion"].motion_dir = "src/assets/motions/your_robot/amp/WalkandRun"
```

The `init_motion_loader` startup event uses `AMPLoader` to load all `.npz` files
from these directories. At each reset, `reset_from_motion` samples a random
frame and writes the root pose, root velocity, and joint state to the simulator.

**Recovery motions** are optional. Set `recovery_dir` only if you want the
delayed-reset mechanism (a subset of envs start from a fallen pose and learn to
stand up). Configure the ratio and delay length:

```python
cfg.events["init_motion_loader"].delay_reset_env_ratio = 0.4   # 40% of envs
cfg.events["init_motion_loader"].max_delay_steps = 250          # ~5s at 50Hz
```

### 3.8 Tuning AMP Hyperparameters

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| `amp_reward_coef` | 0.01 – 1.0 | Scales AMP reward magnitude. Higher = stronger motion imitation. |
| `amp_task_reward_lerp` | 0.0 – 0.9 | Blend ratio with task reward. 0 = pure AMP, 1 = pure task. |
| `amp_discr_hidden_dims` | [1024, 512, 256] | Discriminator capacity. Larger = more expressive but slower. |
| `amp_replay_buffer_size` | 100k – 1M | Policy transitions stored. Larger = more stable training. |
| `amp_num_preload_transitions` | 100k – 500k | Expert transitions preloaded at init. |

**Guidelines:**
- Start with `amp_reward_coef=0.1` and `amp_task_reward_lerp=0.75` (75% task, 25% AMP).
- If the policy ignores motion style, increase `amp_reward_coef` or decrease `task_reward_lerp`.
- If training is unstable, increase `amp_replay_buffer_size` or decrease `amp_reward_coef`.
- The discriminator should not overpower the task reward — watch the `amp_policy_pred`
  and `amp_expert_pred` logs. If both converge to 0, the discriminator is too strong.

---

## Step 4 – Train

### 4.1 Define the RL Config

In `config/your_robot/rl_cfg.py`, create your runner config as a plain dict. The
AMP-specific fields go under `train_cfg["algorithm"]`:

```python
# src/tasks/your_task/config/your_robot/rl_cfg.py

def your_robot_amp_train_cfg() -> dict:
    """Build train_cfg dict for AMP training."""
    return {
        "num_steps_per_env": 24,
        "save_interval": 500,
        "empirical_normalization": False,

        # Observation groups: maps set names to lists of group keys
        "obs_groups": {
            "actor": ["actor"],
            "critic": ["critic"],
        },

        # Actor model
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 256],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },

        # Critic model
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 256],
            "activation": "elu",
        },

        # Algorithm (AMPPPO)
        "algorithm": {
            "class_name": "AMPPPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "learning_rate": 1e-3,
            "max_grad_norm": 1.0,
            "optimizer": "adam",
            "schedule": "adaptive",
            "desired_kl": 0.01,

            # AMP-specific
            "amp_motion_files": "src/assets/motions/your_robot/amp/WalkandRun",
            "amp_body_names": ("pelvis", "left_foot", "right_foot", ...),
            "amp_anchor_name": "torso_link",
            "amp_reward_coef": 0.1,
            "amp_task_reward_lerp": 0.75,
            "amp_discr_hidden_dims": [1024, 512, 256],
            "amp_num_preload_transitions": 200000,
            "min_normalized_std": [0.05] * 29,
        },

        # Logger
        "logger": "tensorboard",
    }
```

> **Note:** The `AMPOnPolicyRunner` constructs `AMPPPO` via its
> `construct_algorithm()` factory, which reads `train_cfg["actor"]`,
> `train_cfg["critic"]`, and `train_cfg["algorithm"]` to build the actor, critic,
> discriminator, and algorithm automatically.

### 4.2 Register the Task

In `config/your_robot/__init__.py`, register with the mjlab task registry:

```python
# src/tasks/your_task/config/your_robot/__init__.py
from mjlab.tasks.registry import register_mjlab_task
from rsl_rl.runners import AMPOnPolicyRunner

from .env_cfgs import your_robot_env_cfg
from .rl_cfg import your_robot_amp_train_cfg

register_mjlab_task(
    task_id="YourRobot-AMP-Rough",
    env_cfg=your_robot_env_cfg(),
    play_env_cfg=your_robot_env_cfg(play=True),
    rl_cfg=your_robot_amp_train_cfg(),
    runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
    task_id="YourRobot-AMP-Flat",
    env_cfg=your_robot_flat_env_cfg(),
    play_env_cfg=your_robot_flat_env_cfg(play=True),
    rl_cfg=your_robot_amp_train_cfg(),
    runner_cls=AMPOnPolicyRunner,
)
```

> **Auto-discovery:** `src/tasks/__init__.py` recursively imports all sub-packages,
> which triggers your `register_mjlab_task()` calls automatically. No manual wiring
> needed — just make sure your package is under `src/tasks/`.

### 4.3 Launch Training

```bash
# List registered tasks to confirm yours appears
python scripts/list_envs.py --keyword YourRobot

# Train
python scripts/train.py YourRobot-AMP-Flat          # flat terrain
python scripts/train.py YourRobot-AMP-Rough          # rough terrain

# Evaluate
python scripts/play.py YourRobot-AMP-Rough \
    --checkpoint-file logs/rsl_rl/your_task/<run_dir>/model_<iter>.pt

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py YourRobot-AMP-Rough
```

Logs and checkpoints go to `logs/rsl_rl/<task_name>/<timestamp>/`.

---

## Appendix A – NPZ Motion Format

The `AMPLoader` (`rsl_rl.utils.motion_loader`) reads all `.npz` files from a
directory and concatenates them into a flat frame buffer.

### Required Keys

| Key              | Type     | Shape               | Description                                      |
|------------------|----------|---------------------|--------------------------------------------------|
| `fps`            | `float`  | scalar              | Frames per second (e.g. 30)                      |
| `joint_pos`      | `float32`| `(T, num_joints)`   | Joint positions in radians                       |
| `joint_vel`      | `float32`| `(T, num_joints)`   | Joint velocities in rad/s                        |
| `body_pos_w`     | `float32`| `(T, num_bodies, 3)`| Body positions in world frame                    |
| `body_quat_w`    | `float32`| `(T, num_bodies, 4)`| Body quaternions in world frame (wxyz)           |
| `body_lin_vel_w` | `float32`| `(T, num_bodies, 3)`| Body linear velocities in world frame            |
| `body_ang_vel_w` | `float32`| `(T, num_bodies, 3)`| Body angular velocities in world frame           |

### How AMPLoader Processes Motion Data

1. **Loading:** All `.npz` files in the directory are loaded and concatenated
   along the time axis into a single flat buffer of `total_frames` frames.

2. **Body selection:** Only the bodies listed in `body_names` are kept (matching
   the `amp_body_names` config). The `anchor_name` body is used as the reference
   frame origin.

3. **Relative computation:** For each frame, the loader computes:
   - `body_pos_b`: body positions relative to anchor, rotated into anchor frame
   - `body_ori_b`: body orientation as 3×3 rotation matrix columns (9 values per body)
   - `body_lin_vel_b`: body linear velocities relative to anchor, in anchor frame
   - `body_ang_vel_b`: body angular velocities relative to anchor, in anchor frame

4. **Observation vector:** These are concatenated into a single vector per frame:
   ```
   [pos_b(3) | ori_b(9) | lin_vel_b(3) | ang_vel_b(3)] × num_bodies
   ```

5. **Pair generation:** `feed_forward_generator(n, batch_size)` samples `n` batches
   of `(state, next_state)` pairs — consecutive frames from random clips. These
   are the expert transitions the discriminator learns to identify.

### Notes

- `T` is the number of frames in a single clip. Multiple clips in the same directory
  are concatenated along the time axis.
- `num_joints` must match the robot's actuated DOFs (used for reset, not for AMP obs).
- `num_bodies` must match the number of bodies in the MJCF model (including the
  root body). The `body_names` list selects which bodies to track.
- The last frame of a clip has `next_state == state` (clamped). This produces a
  zero-velocity transition — a minor data quality issue that doesn't affect training.

---

## Appendix B – ONNX Export

### Auto-Export During Training

The `AMPOnPolicyRunner` inherits ONNX export from the base `OnPolicyRunner`. To
export from a trained checkpoint:

```bash
python scripts/play.py YourRobot-AMP-Rough \
    --checkpoint-file logs/rsl_rl/your_task/<run>/model_30000.pt
```

The runner writes `policy.onnx` alongside the checkpoint (opset 18, external data
inlined).

### Inference Policy

To get the inference policy programmatically:

```python
runner = AMPOnPolicyRunner(env, train_cfg, log_dir, device)
runner.load("path/to/model.pt")
policy = runner.get_inference_policy(device="cuda:0")

# For TensorDict observations:
actions = policy(obs_td)

# For plain tensors (auto-wrapped):
actions = policy(plain_obs_tensor)
```

---

## Appendix C – Rough Terrain (Optional)

To add rough terrain, set the terrain generator on the scene config:

```python
# In your per-robot env_cfgs.py
from src.tasks.amp_loco.mdp.terrain import RANDOM_ROUGH_TERRAINS_CFG

cfg.scene.terrain = RANDOM_ROUGH_TERRAINS_CFG
```

This is already included in the `_rough_env_cfg()` variants. Create a separate
`_flat_env_cfg()` that omits it (or sets `terrain=None`) for flat-ground training.
