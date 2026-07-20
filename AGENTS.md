# Repository Guidelines

## Project Structure & Module Organization

`src/` contains the installable Python package. Keep AMP locomotion and recovery work in `src/tasks/amp_loco/`: environment factories live in `amp_env_cfg.py`, reusable MDP terms in `mdp/`, G1-specific registration and RL settings in `config/g1/`, and AMP runner code in `rl/`. `src/tasks/velocity/` is the base velocity task and a useful reference for shared behavior. Robot definitions are under `src/assets/robots/unitree_g1/`; AMP motion clips are NPZ files in `src/assets/motions/g1/amp/{WalkandRun,Recovery}/`.

Use `scripts/` for runnable tools (`train.py`, `play.py`, and motion conversion). Put raw motion CSVs in `motion_data_csv/amp/`. `mjlab_patch/` holds an optional upstream mjlab patch; do not treat it as application source.

## Build, Test, and Development Commands

Activate the `mjlab` Conda environment, install the AMP-capable `rsl_rl` fork, then install this project:

```bash
python -m pip install -e .
python scripts/list_envs.py --keyword AMP
python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096
python scripts/play.py Unitree-G1-AMP-Rough --checkpoint-file logs/rsl_rl/g1_amp_locomotion/<run>/model_<iter>.pt
python scripts/csv_to_npz.py --help
```

The list command is the fastest registration smoke check. Training produces runs and ONNX exports under `logs/rsl_rl/g1_amp_locomotion/`; use a small environment count when validating changes locally. Playback validates a saved checkpoint. There is currently no committed automated test suite.

## Coding Style & Naming Conventions

Write Python with two-space indentation and match the surrounding file’s formatting. Use `snake_case` for functions, variables, and modules; `PascalCase` for classes and config types; and explicit, typed function signatures for MDP terms. Keep imports grouped as standard library, third-party packages, then local modules. Place task-specific reward, observation, event, and termination logic in the corresponding `mdp/` module rather than embedding it in config files. No formatter or linter is configured—avoid unrelated reformatting.

## Testing Guidelines

For changes to task registration or configuration, run `python scripts/list_envs.py --keyword AMP`. For behavior changes, run a short training or replay appropriate to the affected Flat or Rough task, and report the command, checkpoint/run, and observed result. Keep generated logs, checkpoints, and local datasets out of commits.

## Commit & Pull Request Guidelines

Recent history favors short lowercase subjects, often with a Conventional Commit-style scope, such as `fix(amp_loco): prevent rough-terrain crash` or `refactor: decouple rsl_rl`. Use that style with a focused imperative summary. PRs should describe the affected task/configuration, list validation commands and outcomes, link relevant issues, and include plots or playback screenshots when training behavior changes. Call out motion-data, dependency, or `mjlab_patch` requirements explicitly.
