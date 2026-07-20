# AMP Recovery Tasks

This document describes the Unitree G1 fall-recovery tasks registered by
`src/tasks/amp_loco/config/g1/`.

## Available tasks

| Task | Purpose | Assistance during play |
| --- | --- | --- |
| `Unitree-G1-AMP-Recovery-Flat` | Train recovery-only AMP behavior on flat terrain. | Disabled |
| `Unitree-G1-AMP-Recovery-Flat-Debug` | Inspect recovery behavior and the assistance overlay. | Enabled |

Both tasks use recovery demonstrations from
`src/assets/motions/g1/amp/Recovery`, zero velocity commands, and delayed
termination to provide a recovery window. The normal task is the one to use
for training and unassisted evaluation. The debug task is intended only for
viewer-based verification.

## Training

List the registered tasks:

```bash
python scripts/list_envs.py --keyword Recovery
```

Start recovery-only training with a smaller local rollout first:

```bash
python scripts/train.py Unitree-G1-AMP-Recovery-Flat \
  --env.scene.num-envs=256
```

During training, each recovery environment samples an upward force uniformly
from `[0, 200] N` at reset. The force is applied in world-frame `+Z` to
`torso_link` while delayed recovery is active. Its scale decreases linearly
with the global environment step counter and reaches exactly zero after
`5,000` policy iterations (`5,000 × 24` environment steps). The counter is
stored in checkpoints so resuming training preserves the annealing schedule.

Monitor these logged values in TensorBoard or W&B:

* `Curriculum/upward_assistance_scale`
* `Metrics/upward_assistance_active_fraction`
* `Metrics/upward_assistance_force_mean`

## Playback and force visualization

Evaluate a checkpoint without assistance:

```bash
python scripts/play.py Unitree-G1-AMP-Recovery-Flat \
  --checkpoint-file logs/rsl_rl/g1_amp_recovery/<run>/model_<step>.pt \
  --num-envs 1
```

To inspect the applied force, use the debug task. It draws an orange upward
arrow at the torso center of mass; arrow length is proportional to the current
force (a `200 N` force is approximately `0.4 m`).

```bash
python scripts/play.py Unitree-G1-AMP-Recovery-Flat-Debug \
  --checkpoint-file logs/rsl_rl/g1_amp_recovery/<run>/model_<step>.pt \
  --num-envs 1 --viewer native
```

The arrow appears only while the delayed recovery assistance is active and
disappears when recovery completes or the annealing schedule reaches zero.

See [the integration guide](amp-integration-guide.md) and
[the AMP PPO technical report](amp-ppo-technical-report.md) for broader
architecture and training details.
