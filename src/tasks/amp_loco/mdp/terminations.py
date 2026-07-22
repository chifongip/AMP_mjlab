from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.termination_manager import TerminationManager
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class DelayedTerminationManager(TerminationManager):
    """TerminationManager subclass that delays reset for a subset of envs.

    For delay envs, when a termination is triggered the reset signal is
    suppressed and a counter starts incrementing. Once the counter reaches
    ``max_delay_steps``, the reset signal is released and the counter resets.
    """

    def __init__(
        self,
        base: TerminationManager,
        delay_env_mask: torch.Tensor,
        max_delay_steps: int,
    ) -> None:
        # Steal all internal state from the base manager (avoid re-init).
        self.__dict__.update(base.__dict__)
        self._delay_env_mask = delay_env_mask          # (num_envs,) bool
        self._delay_counters = torch.zeros_like(delay_env_mask, dtype=torch.long)
        self._max_delay_steps = max_delay_steps
        self._delay_failure_buf = torch.zeros_like(delay_env_mask)
        self._delay_timeout_buf = torch.zeros_like(delay_env_mask)

    def compute(self) -> torch.Tensor:
        super().compute()  # fills _truncated_buf, _terminated_buf
        self._delay_failure_buf.zero_()
        self._delay_timeout_buf.copy_(self._delay_env_mask & self._truncated_buf)

        if self._max_delay_steps <= 0:
            return self._truncated_buf | self._terminated_buf

        # Only delay task failures. Timeouts remain immediate episode boundaries.
        delay_and_terminated = self._delay_env_mask & self._terminated_buf
        self._delay_counters[delay_and_terminated] += 1

        # Delay envs whose counter hasn't reached threshold: suppress reset.
        not_ready = delay_and_terminated & (
            self._delay_counters < self._max_delay_steps
        )
        self._terminated_buf[not_ready] = False

        # Expose a one-step pulse before clearing failed attempts for reset.
        ready = delay_and_terminated & (
            self._delay_counters >= self._max_delay_steps
        )
        self._delay_failure_buf.copy_(ready & ~self._delay_timeout_buf)
        self._delay_counters[ready] = 0

        # Clear counters after recovery or an episode timeout.
        recovered = self._delay_env_mask & ~delay_and_terminated
        self._delay_counters[recovered | self._delay_timeout_buf] = 0

        return self._truncated_buf | self._terminated_buf
