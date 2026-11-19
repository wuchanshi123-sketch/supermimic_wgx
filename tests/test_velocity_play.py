"""Playback coverage for the standalone velocity actor and command input."""

import json
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from supermimic.runners.velocity_runner import VelocityActor, VelocityRunnerCfg
from supermimic.scripts.play_velocity import GamepadVelocityPolicy, load_velocity_actor
from supermimic.tasks.velocity_env_cfg import velocity_env_cfg


@pytest.mark.parametrize("with_config", [False, True])
def test_load_local_velocity_actor_with_custom_history(tmp_path, with_config):
  num_actions, sensor_width, history_length = 29, 93, 2
  actor_input = sensor_width * history_length + 3
  cfg = VelocityRunnerCfg(
    actor_hidden_dims=(16, 12),
    action_limit=2.0 if with_config else VelocityRunnerCfg.action_limit,
  )
  original = VelocityActor(actor_input, num_actions, cfg).eval()
  checkpoint = tmp_path / "actor.pt"
  torch.save(original.state_dict(), checkpoint)
  if with_config:
    (tmp_path / "runner_config.json").write_text(
      json.dumps({"runner_cfg": {"action_limit": cfg.action_limit}})
    )

  loaded, history = load_velocity_actor(checkpoint, sensor_width, num_actions, 3, "cpu")
  assert history == history_length
  assert loaded.action_limit == cfg.action_limit
  obs = TensorDict({"actor": torch.randn(1, actor_input)}, [1])
  torch.testing.assert_close(loaded(obs), original(obs))


def test_gamepad_command_reaches_actor_and_environment_before_step():
  env = SimpleNamespace(num_envs=1, device="cpu")
  command = velocity_env_cfg(play=True).commands["twist"].build(env)
  indices = torch.tensor([0])
  command.reset(indices)
  samples = iter([(1.0, -1.0, 1.0), (-1.0, 1.0, -1.0), (0.0, 0.0, 0.0)])
  pad = SimpleNamespace(read=lambda: next(samples))
  policy = GamepadVelocityPolicy(lambda obs: obs["actor"], 2, command, pad)

  def observation(value):
    return TensorDict(
      {"actor": torch.tensor([[value]]), "goal": torch.zeros(1, 3)}, [1]
    )

  torch.testing.assert_close(
    policy(observation(1.0)), torch.tensor([[1.0, 1.0, 1.0, -1.0, 3.0]])
  )
  command.compute(dt=10.0)
  torch.testing.assert_close(command.command, torch.tensor([[1.0, -1.0, 3.0]]))
  policy.reset(torch.tensor([False]))
  torch.testing.assert_close(
    policy(observation(2.0)), torch.tensor([[1.0, 2.0, -1.0, 1.0, -3.0]])
  )
  command.reset(indices)
  policy.reset(torch.tensor([True]))
  torch.testing.assert_close(command.command, torch.zeros(1, 3))
  torch.testing.assert_close(
    policy(observation(3.0)), torch.tensor([[3.0, 3.0, 0.0, 0.0, 0.0]])
  )
