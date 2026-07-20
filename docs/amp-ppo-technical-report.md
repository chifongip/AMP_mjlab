# AMP PPO: Adversarial Motion Priors with Proximal Policy Optimization

**Technical Report — AMP_mjlab Implementation**

---

## 1. Introduction

AMP PPO combines two techniques for training humanoid locomotion policies:

- **PPO (Proximal Policy Optimization)** — a standard reinforcement learning algorithm that learns by trial and error, updating the policy in small safe steps.
- **AMP (Adversarial Motion Priors)** — a technique that uses motion capture data to teach a robot *how* to move, without hand-crafting reward functions for every aspect of natural motion.

The key insight: instead of designing dozens of reward terms to describe "natural walking," we show the robot example motions and let a learned discriminator judge whether its behavior looks like the examples. This discriminator score becomes the reward signal.

---

## 2. Architecture Overview

AMP PPO has three neural networks and a statistical normalizer:

| Component | Input | Output | Role |
|-----------|-------|--------|------|
| **Actor** | Robot state (proprioception) | Joint actions | Decides what to do |
| **Critic** | Robot state + extra observations | Value estimate | Judges how good the current state is |
| **Discriminator** | (current AMP obs, next AMP obs) | Scalar logit | Judges whether motion looks like the expert |
| **Observation Normalizer** | Raw observations | Normalized observations | Running mean/std (not learned) |

The **discriminator** is the AMP-specific addition. It takes *pairs* of consecutive AMP observations — capturing motion *transitions*, not just poses — and outputs a score indicating how "expert-like" the motion is.

### AMP Observation Space

Each AMP observation encodes the robot's full body state relative to its root (pelvis):

```
Per body: [3D position, 6D orientation, 3D linear velocity, 3D angular velocity]
Total:    15 features × number of bodies
```

All quantities are expressed in the root body's local frame, making the representation invariant to global position and heading.

---

## 3. Training Loop

The training loop alternates between **rollout collection** (acting in the environment) and **policy updates** (learning from collected data).

### Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING ITERATION                           │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    ROLLOUT PHASE                              │  │
│  │                                                               │  │
│  │   ┌──────────┐    ┌──────────┐    ┌────────────────────────┐  │  │
│  │   │ Observe  │───>│  Actor   │───>│  Environment Step      │  │  │
│  │   │ (obs_td) │    │ (policy) │    │  (physics simulation)  │  │  │
│  │   └──────────┘    └──────────┘    └───────────┬────────────┘  │  │
│  │        ▲                                       │              │  │
│  │        │                                       ▼              │  │
│  │        │                          ┌────────────────────────┐  │  │
│  │        │                          │  Terminal Handling     │  │  │
│  │        │                          │  (fix reset boundary)  │  │  │
│  │        │                          └───────────┬────────────┘  │  │
│  │        │                                      │               │  │
│  │        │                                      ▼               │  │
│  │        │                          ┌────────────────────────┐  │  │
│  │        │                          │  Discriminator         │  │  │
│  │        │                          │  predict_amp_reward()  │  │  │
│  │        │                          │                        │  │  │
│  │        │                          │  d = D(s, s')          │  │  │
│  │        │                          │  r_amp = clamp(1 - .25 │  │  │
│  │        │                          │       * (d-1)², 0)     │  │  │
│  │        │                          │  r = lerp(r_amp, r_task)│ │  │
│  │        │                          └───────────┬────────────┘  │  │
│  │        │                                      │               │  │
│  │        │                                      ▼               │  │
│  │        │                         ┌────────────────────────┐   │  │
│  │        │     ┌──────────────────>│  Store Transition      │   │  │
│  │        │     │                   │  - RolloutStorage      │   │  │
│  │        │     │                   │  - AMP ReplayBuffer    │   │  │
│  │        │     │                   └────────────────────────┘   │  │
│  │        │     │                                                │  │
│  │   Repeat for num_steps_per_env                                │  │
│  └───────┼─────┼─────────────────────────────────────────────────┘  │
│          │     │                                                    │
│          │     ▼                                                    │
│  ┌───────┼────────────────────────────────────────────────────────┐ │
│  │       │           UPDATE PHASE                                 │ │
│  │       │                                                        │ │
│  │       │   ┌──────────────────────────────────────────────┐     │ │
│  │       │   │  For each mini-batch:                        │     │ │
│  │       │   │                                              │     │ │
│  │       │   │  1. PPO Losses (from RolloutStorage)         │     │ │
│  │       │   │     - Surrogate loss (clipped ratio)         │     │ │
│  │       │   │     - Value loss (MSE on returns)            │     │ │
│  │       │   │     - Entropy bonus                          │     │ │
│  │       │   │                                              │     │ │
│  │       │   │  2. AMP Discriminator Loss                   │     │ │
│  │       │   │     - Policy pairs from ReplayBuffer         │     │ │
│  │       │   │     - Expert pairs from AMPLoader            │     │ │
│  │       │   │     - LSGAN loss + gradient penalty          │     │ │
│  │       │   │                                              │     │ │
│  │       │   │  3. Combined backward + optimizer step       │     │ │
│  │       │   └──────────────────────────────────────────────┘     │ │
│  │       │                                                        │ │
│  └───────┼────────────────────────────────────────────────────────┘ │
│          │                                                          │
│          └──── Repeat for num_learning_iterations                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. How Each Component Works

### 4.1 The Actor (Policy Network)

The actor is a standard MLP that maps robot observations to actions:

```
Observation (proprioception) → MLP → [mean, log_std] → sample action
```

- **Input**: Base angular velocity, projected gravity, velocity command, joint positions, joint velocities, previous actions — with a 4-step history.
- **Output**: Gaussian distribution over joint actions. Actions are sampled during training and taken as the mean during inference.
- **Min-std clamping**: After each update, the actor's standard deviation is clamped to a minimum value to maintain exploration.

### 4.2 The Critic (Value Network)

The critic estimates the expected cumulative reward from the current state:

```
Observation (privileged) → MLP → scalar value V(s)
```

- **Input**: Everything the actor sees, plus base linear velocity and body pose in the root frame (extra observations the actor doesn't have access to).
- **Output**: A single scalar estimating how good the current state is.
- **Used for**: Computing GAE (Generalized Advantage Estimation) returns, which tell the actor whether its actions were better or worse than expected.

### 4.3 The Discriminator

The discriminator is the heart of AMP. It learns to distinguish between:

- **Expert transitions**: Consecutive frames from motion capture data (real walking/running)
- **Policy transitions**: Consecutive frames from the robot's own behavior

```
Input:  [AMP_obs_current, AMP_obs_next]  (concatenated)
        ↓
Trunk:  Linear → ReLU → Linear → ReLU → ...
        ↓
Head:   Linear → scalar logit d
```

**Training loss (LSGAN formulation):**

```python
expert_loss  = MSE(d_expert, 1)      # expert should score 1
policy_loss  = MSE(d_policy, -1)     # policy should score -1
disc_loss    = 0.5 * (expert_loss + policy_loss)
```

**Gradient penalty (WGAN-GP style):**

```python
# Encourage smooth discriminator by penalizing large gradients
grad = autograd.grad(d_expert, expert_input)
grad_penalty = lambda * ||grad||²
```

This prevents the discriminator from becoming too confident or too sharp, which would produce uninformative gradients for the actor.

### 4.4 AMP Reward Computation

During rollouts, the discriminator produces an AMP reward that is blended with the environment's task reward:

```python
d = discriminator(amp_obs, next_amp_obs)    # scalar logit
amp_reward = amp_reward_coef * clamp(1 - 0.25 * (d - 1)², min=0)
reward = (1 - task_reward_lerp) * amp_reward + task_reward_lerp * task_reward
```

With `task_reward_lerp = 0.75`, the final reward is 75% task reward (velocity tracking, penalties) and 25% AMP discriminator reward (motion quality).

The AMP component is a **truncated squared reward** from the AMP paper:

| Discriminator output `d` | AMP Reward |
|--------------------------|--------|
| `d = 1` (expert-like) | Maximum |
| `d = -1` or `d = 3` | Zero |
| `d < -1` or `d > 3` | Clamped to zero |

The AMP reward is highest when the discriminator thinks the motion is indistinguishable from the expert (`d ≈ 1`).

### 4.5 The Replay Buffer

A circular buffer storing `(amp_obs, next_amp_obs)` pairs from the policy's recent experience:

```
┌─────────────────────────────────────────────┐
│  ReplayBuffer (capacity: 100,000)           │
│                                             │
│  [s₁, s₁'] [s₂, s₂'] [s₃, s₃'] ...       │
│                                             │
│  insert(): append new pairs (circular)      │
│  sample():  random mini-batch for training  │
└─────────────────────────────────────────────┘
```

This provides **off-policy** data for the discriminator — even though the actor has moved on, the discriminator still trains on recent policy data alongside fresh expert data.

### 4.6 The Motion Loader (Expert Data)

Loads `.npz` motion capture files and preprocesses them into the AMP observation format:

```
Raw motion data (world frame):
  body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w
        ↓
Preprocessing (per frame):
  1. Compute body pose relative to root body
  2. Convert quaternion → 6D rotation (first 2 columns of rotation matrix)
  3. Transform velocities into root body's local frame
        ↓
AMP observation format:
  [pos_b(3), ori_b(6), lin_vel_b(3), ang_vel_b(3)] × num_bodies
```

During training, the loader provides random consecutive-frame pairs `(s, s')` from the expert data — these represent what "natural motion" looks like.

---

## 5. The Update Step in Detail

Each update iterates over mini-batches. Three data generators are zipped together:

```python
for ppo_batch, policy_amp, expert_amp in zip(
    rollout_storage.feed_forward_generator(),    # PPO transitions
    amp_replay_buffer.feed_forward_generator(),   # Policy AMP pairs
    amp_data.feed_forward_generator(),            # Expert AMP pairs
):
```

### 5.1 PPO Losses (Standard)

**Surrogate loss** — the core PPO objective:

```python
ratio = exp(log_pi_new - log_pi_old)
surr1 = ratio * advantage
surr2 = clamp(ratio, 1-ε, 1+ε) * advantage
surrogate_loss = -min(surr1, surr2).mean()
```

This prevents the policy from changing too much in a single update. The clipping parameter `ε` (typically 0.2) limits how far the new policy can deviate from the old one.

**Value loss:**

```python
value_loss = MSE(predicted_value, returns)
```

**Entropy bonus:**

```python
entropy_loss = -entropy.mean()  # Encourages exploration
```

**Adaptive learning rate:**

```python
if kl_divergence < desired_kl / 2:
    lr *= 2    # Policy changing too slowly, speed up
elif kl_divergence > desired_kl * 2:
    lr /= 2    # Policy changing too fast, slow down
```

### 5.2 AMP Discriminator Loss

```python
# Normalize AMP observations
policy_state = amp_normalizer.normalize(policy_state)
expert_state = amp_normalizer.normalize(expert_state)

# Forward pass
d_policy = discriminator(concat(policy_state, policy_next_state))
d_expert = discriminator(concat(expert_state, expert_next_state))

# LSGAN loss
disc_loss = 0.5 * (MSE(d_expert, 1) + MSE(d_policy, -1))

# Gradient penalty
grad_penalty = lambda * ||∇D(expert_input)||²

# Total
total_loss += disc_loss + grad_penalty
```

### 5.3 Combined Optimization

All three networks (actor, critic, discriminator) share a single Adam optimizer but with different weight decay:

| Component | Weight Decay |
|-----------|-------------|
| Actor | 0 |
| Critic | 0 |
| Discriminator trunk | 10⁻³ (0.001) |
| Discriminator head | 10⁻¹ (0.1) |

Gradients are clipped globally across all parameters by `max_grad_norm`.

After the optimizer step, the AMP normalizer's running statistics are updated with the raw (pre-normalization) observations from this mini-batch.

---

## 6. Terminal State Handling

When an environment resets (robot falls, episode timeout), the simulator auto-resets before computing the next observation. This creates a problem: the "next" AMP observation belongs to a *new* episode, not the current one.

The runner fixes this by substituting the pre-reset observation:

```python
# For environments that just reset:
next_amp_obs[reset_env_ids] = amp_obs[reset_env_ids]

# Now the AMP transition (s, s') doesn't cross episode boundaries
# reward = (1-lerp) * amp_reward + lerp * task_reward
rewards = discriminator.predict_amp_reward(amp_obs, next_amp_obs, task_rewards, ...)
```

This ensures the discriminator only sees valid within-episode transitions.

---

## 7. Complete Data Flow

```
Motion Capture (.npz files)
        │
        ▼
AMPLoader ──────────────────────────────────┐
  - Preprocesses to body-local coords       │
  - Provides expert (s, s') pairs           │
                                            │
                                            ▼
Environment ──> Actor ──> Actions ──> Physics Step
    │                                       │
    │ AMP obs (s)                           │ AMP obs (s')
    │                                       │
    ▼                                       ▼
Discriminator.predict_amp_reward(s, s', task_reward)
    │
    │ r = (1-lerp) * amp_reward + lerp * task_reward
    ▼
RolloutStorage ← store (obs, action, reward, value, ...)
ReplayBuffer   ← store (amp_obs, next_amp_obs)
    │
    ▼
GAE.compute_returns()  ← TD-lambda on AMP rewards
    │
    ▼
AMPPPO.update()
    ├── PPO loss (from RolloutStorage)
    └── Discriminator loss (from ReplayBuffer + AMPLoader)
         ├── LSGAN: MSE(d_expert, 1) + MSE(d_policy, -1)
         └── Gradient penalty: λ||∇D||²
```

---

## 8. Key Design Decisions

**Why mix AMP and task rewards?**
The AMP reward captures "move like the expert" which implicitly encodes natural gait and smooth motion. However, pure AMP reward can struggle with task-specific objectives like velocity tracking. The `task_reward_lerp` parameter (0.75 in this project) blends task rewards back in: `(1 - lerp) * amp_reward + lerp * task_reward`. This gives the task reward dominance while the AMP discriminator regularizes motion quality.

**Why use observation pairs (s, s') instead of single states?**
A single pose can't distinguish between "standing still" and "mid-stride." Consecutive frames capture the *dynamics* — the velocity and acceleration patterns that define natural motion.

**Why body-local coordinates?**
Expressing everything relative to the root body makes the observation invariant to global position and heading. The discriminator learns "how the legs move relative to the torso" rather than "where the robot is in the world."

**Why a replay buffer for policy data?**
The discriminator needs to see both policy and expert data in each mini-batch. Without a buffer, you'd only have the current rollout's data. The buffer provides a longer history of policy behavior, stabilizing discriminator training.

**Why LSGAN instead of standard GAN?**
Standard GAN (binary cross-entropy) can suffer from vanishing gradients when the discriminator is confident. LSGAN (MSE loss) provides smoother gradients, which helps the actor learn from the discriminator's feedback.

---

## 9. Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `amp_reward_coef` | 0.1 | Scales the AMP reward magnitude |
| `amp_task_reward_lerp` | 0.75 | Fraction of task reward mixed in (0 = pure AMP, 1 = pure task) |
| `amp_discr_hidden_dims` | [1024, 512, 256] | Discriminator network architecture |
| Gradient penalty λ | 10 | Lipschitz constraint strength |
| `num_learning_epochs` | 5 | Update epochs per iteration |
| `num_mini_batches` | 4 | Mini-batches per epoch |
| `clip_param` | 0.2 | PPO clipping parameter |
| `desired_kl` | 0.01 | Target KL divergence for adaptive LR |
| Replay buffer size | 100,000 | Policy transition history |
