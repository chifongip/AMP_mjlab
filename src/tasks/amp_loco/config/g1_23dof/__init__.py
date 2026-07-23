from mjlab.tasks.registry import register_mjlab_task

from src.tasks.amp_loco.rl import AMPOnPolicyRunner

from .env_cfgs import g1_23dof_amp_recovery_flat_env_cfg
from .rl_cfg import g1_23dof_amp_recovery_ppo_runner_cfg


register_mjlab_task(
  task_id="Unitree-G1-23Dof-AMP-Recovery-Flat",
  env_cfg=g1_23dof_amp_recovery_flat_env_cfg(),
  play_env_cfg=g1_23dof_amp_recovery_flat_env_cfg(play=True),
  rl_cfg=g1_23dof_amp_recovery_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

# Verification-only task: keep assistance enabled while using the play viewer.
register_mjlab_task(
  task_id="Unitree-G1-23Dof-AMP-Recovery-Flat-Debug",
  env_cfg=g1_23dof_amp_recovery_flat_env_cfg(),
  play_env_cfg=g1_23dof_amp_recovery_flat_env_cfg(),
  rl_cfg=g1_23dof_amp_recovery_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)
