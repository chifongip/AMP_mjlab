import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mjlab.managers.termination_manager import TerminationManager

from src.tasks.amp_loco.recovery_curriculum import (
  configure_recovery_curriculum_interval,
  validate_recovery_curriculum_workers,
)
from src.tasks.amp_loco.mdp.events import (
  AssistanceCurriculum,
  UpwardRecoveryAssistance,
)
from src.tasks.amp_loco.mdp.metrics import (
  recovery_failure_rate,
  recovery_timeout_rate,
)
from src.tasks.amp_loco.mdp.terminations import DelayedTerminationManager


def make_assistance_state(
  curriculum: AssistanceCurriculum,
  num_envs: int = 4,
) -> UpwardRecoveryAssistance:
  assistance = UpwardRecoveryAssistance.__new__(UpwardRecoveryAssistance)
  assistance._curriculum = curriculum
  assistance._device = "cpu"
  assistance._num_envs = num_envs
  assistance._success_ema = None
  assistance._last_evaluation_step = 0
  assistance._last_attempt_success_rate = None
  assistance._last_attempt_failure_rate = None
  assistance._last_attempt_timeout_rate = None
  assistance._attempt_active = torch.zeros(num_envs, dtype=torch.bool)
  assistance._stable_counts = torch.zeros(num_envs, dtype=torch.long)
  assistance._attempt_steps = torch.zeros(num_envs, dtype=torch.long)
  for name in (
    "_pending_successes",
    "_pending_failures",
    "_pending_timeouts",
    "_completed_successes",
    "_completed_failures",
    "_completed_timeouts",
    "_completed_success_steps",
  ):
    setattr(assistance, name, torch.zeros((), dtype=torch.long))
  return assistance


class AssistanceCurriculumTest(unittest.TestCase):
  def make_curriculum(self) -> AssistanceCurriculum:
    return AssistanceCurriculum(
      scales=(1.0, 0.75, 0.5, 0.25, 0.0),
      advance_threshold=0.8,
      advance_hold_evaluations=3,
      rollback_threshold=0.6,
      rollback_hold_evaluations=2,
    )

  def test_advances_only_after_sustained_success(self) -> None:
    curriculum = self.make_curriculum()
    curriculum.update(0.9)
    curriculum.update(0.9)
    self.assertEqual(curriculum.stage, 0)
    curriculum.update(0.9)
    self.assertEqual(curriculum.stage, 1)
    self.assertEqual(curriculum.scale, 0.75)

  def test_pauses_in_the_hysteresis_band(self) -> None:
    curriculum = self.make_curriculum()
    curriculum.stage = 2
    for _ in range(10):
      curriculum.update(0.7)
    self.assertEqual(curriculum.stage, 2)

  def test_rolls_back_after_sustained_regression(self) -> None:
    curriculum = self.make_curriculum()
    curriculum.stage = 4
    curriculum.update(0.5)
    self.assertEqual(curriculum.stage, 4)
    curriculum.update(0.5)
    self.assertEqual(curriculum.stage, 3)
    self.assertEqual(curriculum.scale, 0.25)

  def test_rejects_invalid_scales(self) -> None:
    with self.assertRaises(ValueError):
      AssistanceCurriculum(
        scales=(1.0, 0.5),
        advance_threshold=0.8,
        advance_hold_evaluations=1,
        rollback_threshold=0.6,
        rollback_hold_evaluations=1,
      )

  def test_assistance_state_round_trip(self) -> None:
    source = make_assistance_state(self.make_curriculum())
    source._curriculum.stage = 3
    source._curriculum.advance_count = 2
    source._curriculum.rollback_count = 1
    source._success_ema = 0.84
    source._last_evaluation_step = 2400
    source._last_attempt_success_rate = 0.81
    source._pending_successes.fill_(11)
    source._pending_failures.fill_(3)
    source._completed_successes.fill_(41)
    source._completed_failures.fill_(9)
    source._completed_success_steps.fill_(820)

    restored = make_assistance_state(self.make_curriculum())
    restored.load_state_dict(source.state_dict())

    self.assertEqual(restored.state_dict(), source.state_dict())

  def test_v1_checkpoint_discards_occupancy_evidence(self) -> None:
    restored = make_assistance_state(self.make_curriculum())
    restored._env = SimpleNamespace(common_step_counter=2400)

    restored.load_state_dict(
      {
        "stage": 2,
        "advance_count": 99,
        "rollback_count": 10,
        "success_ema": 0.91,
        "last_evaluation_step": 2377,
      }
    )

    self.assertEqual(restored._curriculum.stage, 2)
    self.assertEqual(restored._curriculum.advance_count, 0)
    self.assertEqual(restored._curriculum.rollback_count, 0)
    self.assertIsNone(restored._success_ema)
    self.assertEqual(restored._last_evaluation_step, 2400)


class RecoveryAttemptTrackingTest(unittest.TestCase):
  def make_assistance(self, num_envs: int = 3) -> UpwardRecoveryAssistance:
    curriculum = AssistanceCurriculum(
      scales=(1.0, 0.5, 0.0),
      advance_threshold=0.8,
      advance_hold_evaluations=2,
      rollback_threshold=0.5,
      rollback_hold_evaluations=2,
    )
    assistance = make_assistance_state(curriculum, num_envs)
    assistance._minimum_height = 0.7
    assistance._maximum_tilt = torch.deg2rad(torch.tensor(20.0)).item()
    assistance._stable_steps_required = 2
    assistance._minimum_completed_attempts = 2
    assistance._success_ema_alpha = 0.5
    assistance._body_ids = [0]

    root_pos = torch.zeros(num_envs, 1, 3)
    root_pos[:, 0, 2] = 0.8
    root_quat = torch.zeros(num_envs, 1, 4)
    root_quat[:, 0, 0] = 1.0
    assistance._asset = SimpleNamespace(
      data=SimpleNamespace(
        body_link_pos_w=root_pos,
        body_link_quat_w=root_quat,
        gravity_vec_w=torch.tensor([[0.0, 0.0, -1.0]]).repeat(
          num_envs, 1
        ),
      )
    )
    return assistance

  def make_env(self, assistance: UpwardRecoveryAssistance) -> SimpleNamespace:
    return SimpleNamespace(
      scene=SimpleNamespace(env_origins=torch.zeros(assistance._num_envs, 3))
    )

  def test_stable_success_is_counted_once(self) -> None:
    assistance = self.make_assistance()
    assistance._attempt_active[:] = True
    tm = SimpleNamespace(_delay_env_mask=torch.ones(3, dtype=torch.bool))
    env = self.make_env(assistance)

    assistance._update_attempts(env, tm)
    self.assertEqual(int(assistance._pending_successes), 0)
    assistance._update_attempts(env, tm)
    self.assertEqual(int(assistance._pending_successes), 3)
    self.assertFalse(torch.any(assistance._attempt_active))

    assistance._update_attempts(env, tm)
    self.assertEqual(int(assistance._pending_successes), 3)

  def test_stability_counter_resets_on_posture_regression(self) -> None:
    assistance = self.make_assistance(num_envs=1)
    assistance._attempt_active[:] = True
    tm = SimpleNamespace(_delay_env_mask=torch.ones(1, dtype=torch.bool))
    env = self.make_env(assistance)

    assistance._update_attempts(env, tm)
    assistance._asset.data.body_link_pos_w[:, 0, 2] = 0.4
    assistance._update_attempts(env, tm)
    assistance._asset.data.body_link_pos_w[:, 0, 2] = 0.8
    assistance._update_attempts(env, tm)
    self.assertEqual(int(assistance._pending_successes), 0)
    assistance._update_attempts(env, tm)
    self.assertEqual(int(assistance._pending_successes), 1)

  def test_reset_records_one_outcome_and_starts_new_attempt(self) -> None:
    assistance = self.make_assistance()
    assistance._attempt_active[:] = True
    tm = SimpleNamespace(
      _delay_env_mask=torch.tensor([True, True, False]),
      _delay_failure_buf=torch.tensor([True, False, False]),
      _delay_timeout_buf=torch.tensor([False, True, False]),
    )
    assistance._env = SimpleNamespace(termination_manager=tm)

    assistance._record_reset_outcomes(torch.arange(3))

    self.assertEqual(int(assistance._pending_failures), 1)
    self.assertEqual(int(assistance._pending_timeouts), 1)
    torch.testing.assert_close(
      assistance._attempt_active,
      torch.tensor([True, True, False]),
    )

  def test_timeout_after_success_is_not_a_second_outcome(self) -> None:
    assistance = self.make_assistance(num_envs=1)
    assistance._attempt_active[:] = False
    tm = SimpleNamespace(
      _delay_env_mask=torch.tensor([True]),
      _delay_failure_buf=torch.tensor([False]),
      _delay_timeout_buf=torch.tensor([True]),
    )
    assistance._env = SimpleNamespace(termination_manager=tm)

    assistance._record_reset_outcomes(torch.tensor([0]))

    self.assertEqual(int(assistance._pending_timeouts), 0)
    self.assertTrue(assistance._attempt_active[0])

  def test_curriculum_uses_completed_attempt_rate(self) -> None:
    assistance = self.make_assistance()
    assistance._pending_successes.fill_(1)
    assistance._pending_failures.fill_(1)

    self.assertTrue(assistance._evaluate_curriculum(24))
    self.assertEqual(assistance._last_attempt_success_rate, 0.5)
    self.assertEqual(assistance._curriculum.stage, 0)
    self.assertEqual(int(assistance._pending_attempts()), 0)

    self.assertFalse(assistance._evaluate_curriculum(48))
    self.assertEqual(assistance._curriculum.rollback_count, 0)

  def test_stage_transition_clears_prior_stage_ema(self) -> None:
    assistance = self.make_assistance()
    assistance._curriculum.advance_hold_evaluations = 1
    assistance._pending_successes.fill_(2)

    assistance._evaluate_curriculum(24)

    self.assertEqual(assistance._curriculum.stage, 1)
    self.assertIsNone(assistance._success_ema)


class DelayedTerminationOutcomeTest(unittest.TestCase):
  def make_manager(self) -> DelayedTerminationManager:
    manager = DelayedTerminationManager.__new__(DelayedTerminationManager)
    manager._delay_env_mask = torch.ones(3, dtype=torch.bool)
    manager._delay_counters = torch.tensor([1, 1, 0])
    manager._max_delay_steps = 2
    manager._delay_failure_buf = torch.zeros(3, dtype=torch.bool)
    manager._delay_timeout_buf = torch.zeros(3, dtype=torch.bool)
    manager._terminated_buf = torch.tensor([True, True, False])
    manager._truncated_buf = torch.tensor([False, True, True])
    return manager

  def test_exposes_distinct_failure_and_timeout_pulses(self) -> None:
    manager = self.make_manager()
    with patch.object(
      TerminationManager,
      "compute",
      return_value=torch.ones(3, dtype=torch.bool),
    ):
      dones = manager.compute()

    torch.testing.assert_close(
      manager._delay_failure_buf,
      torch.tensor([True, False, False]),
    )
    torch.testing.assert_close(
      manager._delay_timeout_buf,
      torch.tensor([False, True, True]),
    )
    self.assertTrue(torch.all(dones))


class RecoveryTrainingConfigurationTest(unittest.TestCase):
  def test_rollout_length_drives_curriculum_interval(self) -> None:
    assistance_cfg = SimpleNamespace(
      params={"evaluation_interval_steps": 24}
    )
    events = {"upward_recovery_assistance": assistance_cfg}

    self.assertTrue(configure_recovery_curriculum_interval(events, 96))
    self.assertEqual(assistance_cfg.params["evaluation_interval_steps"], 96)

  def test_distributed_recovery_curriculum_is_rejected(self) -> None:
    events = {"upward_recovery_assistance": SimpleNamespace()}

    with self.assertRaisesRegex(ValueError, "single-GPU"):
      validate_recovery_curriculum_workers(events, 2)

    validate_recovery_curriculum_workers(events, 1)


class RecoveryCohortMetricsTest(unittest.TestCase):
  def make_env(self) -> SimpleNamespace:
    termination_manager = SimpleNamespace(
      _delay_env_mask=torch.tensor([True, True, False, False]),
      _delay_counters=torch.tensor([3, 0, 5, 5]),
      time_outs=torch.tensor([False, True, True, True]),
    )
    return SimpleNamespace(
      num_envs=4,
      device="cpu",
      termination_manager=termination_manager,
    )

  def test_failure_rate_excludes_home_environments(self) -> None:
    values = recovery_failure_rate(self.make_env())
    torch.testing.assert_close(values, torch.full((4,), 0.5))

  def test_timeout_rate_excludes_home_environments(self) -> None:
    values = recovery_timeout_rate(self.make_env())
    torch.testing.assert_close(values, torch.full((4,), 0.5))


if __name__ == "__main__":
  unittest.main()
