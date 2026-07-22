from __future__ import annotations

from typing import Any


RECOVERY_ASSISTANCE_EVENT = "upward_recovery_assistance"


def configure_recovery_curriculum_interval(
  events: dict[str, Any],
  rollout_steps: int,
) -> bool:
  """Align recovery curriculum evaluation with the PPO rollout boundary."""
  assistance_cfg = events.get(RECOVERY_ASSISTANCE_EVENT)
  if assistance_cfg is None:
    return False
  if rollout_steps <= 0:
    raise ValueError("rollout_steps must be positive.")
  assistance_cfg.params["evaluation_interval_steps"] = rollout_steps
  return True


def validate_recovery_curriculum_workers(
  events: dict[str, Any],
  num_workers: int,
) -> None:
  """Reject distributed recovery curricula until state synchronization exists."""
  if num_workers > 1 and RECOVERY_ASSISTANCE_EVENT in events:
    raise ValueError(
      "Recovery assistance curriculum currently supports single-GPU training "
      "only; choose one GPU until curriculum state synchronization is added."
    )
