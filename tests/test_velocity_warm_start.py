"""Regression coverage for starting a new velocity run from local weights."""

from dataclasses import replace

import pytest
import torch

from supermimic.runners.velocity_runner import (
  VELOCITY_TASK_ID,
  VelocityRunner,
  VelocityRunnerCfg,
)
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.velocity_env_cfg import velocity_env_cfg


def test_init_checkpoint_loads_actor_and_dynamics_for_warmup(tmp_path):
  cfg = VelocityRunnerCfg(
    actor_hidden_dims=(8,),
    critic_hidden_dims=(8,),
    dynamics_hidden_dims=(8,),
    buffer_size=8,
    exploration_noise=0.0,
  )
  env = create_env(velocity_env_cfg(), num_envs=2, device="cpu")
  try:
    source = VelocityRunner(env, cfg, VELOCITY_TASK_ID, device="cpu")
    actor = source.policy_learner.actor
    critic = source.policy_learner.critic
    with torch.no_grad():
      actor.network[-1].weight.zero_()
      actor.network[-1].bias.fill_(0.25)
      actor.normalizer.update(torch.ones(2, actor.network[0].in_features))
      source._dynamics.mlp[-1].bias.fill_(0.5)
      source._reward_signals.network[-1].bias.fill_(0.3)
      for parameter in critic.parameters():
        parameter.fill_(7.0)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint_dir = run_dir / "checkpoint_42000"
    source._save_checkpoint(checkpoint_dir)

    warm = VelocityRunner(
      env, replace(cfg, init_checkpoint=str(run_dir)), VELOCITY_TASK_ID, device="cpu"
    )
    assert warm._init_actor_path == checkpoint_dir / "actor.pt"
    torch.testing.assert_close(
      warm.policy_learner.actor.state_dict(), actor.state_dict(), rtol=0, atol=0
    )
    torch.testing.assert_close(warm._wm.bundle(), source._wm.bundle(), rtol=0, atol=0)
    torch.testing.assert_close(
      warm._reward_signals.state_dict(),
      source._reward_signals.state_dict(),
      rtol=0,
      atol=0,
    )
    assert not torch.equal(warm.policy_learner.critic[-1].bias, critic[-1].bias)

    assert warm._num_wm_updates == 0
    assert warm.policy_warmup_updates > 0
    assert warm.step() == env.num_envs
    action = warm.replay_buffer.sample(1, env.num_envs)["action"]
    expected = cfg.action_limit * torch.tanh(torch.tensor(0.25))
    torch.testing.assert_close(action, expected.expand_as(action))
  finally:
    env.close()


def test_init_checkpoint_requires_both_weight_files(tmp_path):
  (tmp_path / "actor.pt").touch()
  runner = VelocityRunner.__new__(VelocityRunner)
  runner.cfg = VelocityRunnerCfg(init_checkpoint=str(tmp_path))

  with pytest.raises(FileNotFoundError, match="world_model.pt"):
    runner._initial_actor_path()


def test_init_checkpoint_conflicts_with_wandb_initialization():
  with pytest.raises(ValueError, match="init_checkpoint or wandb_run_path"):
    VelocityRunnerCfg(
      init_checkpoint="checkpoint_42000", wandb_run_path="local/project/run"
    )


def test_init_checkpoint_rejects_a_different_actor_action_limit(tmp_path):
  for name in ("actor.pt", "world_model.pt"):
    (tmp_path / name).touch()
  (tmp_path / "runner_config.json").write_text(
    '{"runner_cfg":{"action_limit":2.0}}', encoding="utf-8"
  )
  runner = VelocityRunner.__new__(VelocityRunner)
  runner.cfg = VelocityRunnerCfg(init_checkpoint=str(tmp_path), action_limit=3.0)

  with pytest.raises(ValueError, match="saved actor uses --runner.action-limit 2"):
    runner._initial_actor_path()
