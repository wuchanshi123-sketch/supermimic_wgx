"""Playback coverage for the standalone velocity actor and command input."""

import json
from types import SimpleNamespace

import mujoco
import pytest
import torch
from tensordict import TensorDict

from supermimic.runners.velocity_runner import VelocityActor, VelocityRunnerCfg
from supermimic.scripts.play_velocity import (
  GamepadVelocityPolicy,
  VelocityWorldModelGhost,
  load_velocity_actor,
)
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


@pytest.mark.parametrize("with_ghost", [False, True])
def test_gamepad_command_reaches_actor_and_environment_before_step(with_ghost):
  env = SimpleNamespace(num_envs=1, device="cpu")
  command = velocity_env_cfg(play=True).commands["twist"].build(env)
  indices = torch.tensor([0])
  command.reset(indices)
  samples = iter([(1.0, -1.0, 1.0), (-1.0, 1.0, -1.0), (0.0, 0.0, 0.0)])
  pad = SimpleNamespace(read=lambda: next(samples))
  calls, resets = [], []

  def step(full_state, command, real_history, real_action):
    calls.append(command.clone())
    torch.testing.assert_close(real_action[:, -3:], command)
    torch.testing.assert_close(full_state, real_history[:, -1] * 10)

  ghost = SimpleNamespace(step=step, reset=lambda: resets.append(True))
  policy = GamepadVelocityPolicy(
    lambda obs: obs["actor"], 2, command, pad, ghost=ghost if with_ghost else None
  )

  def observation(value):
    fields = {"actor": torch.tensor([[value]]), "goal": torch.zeros(1, 3)}
    if with_ghost:
      fields["full_state"] = torch.tensor([[value * 10]])
    return TensorDict(fields, [1])

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
  if with_ghost:
    torch.testing.assert_close(
      torch.cat(calls),
      torch.tensor([[1.0, -1.0, 3.0], [-1.0, 1.0, -3.0], [0.0, 0.0, 0.0]]),
    )
    assert resets == [True]


def test_world_model_ghost_rolls_forward_and_reseeds_without_moving_real_robot():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <body><joint/><geom type="sphere" size="0.1"/></body>
      <body><freejoint/><geom type="sphere" size="0.1"/>
        <body><joint/><geom type="sphere" size="0.1"/></body>
      </body>
    </worldbody></mujoco>
  """)
  robot = SimpleNamespace(
    indexing=SimpleNamespace(
      free_joint_q_adr=torch.arange(1, 8), joint_q_adr=torch.tensor([8])
    )
  )

  class Scene(dict):
    env_origins = torch.tensor([[10.0, 20.0, 30.0]])

  env = SimpleNamespace(scene=Scene(robot=robot), sim=SimpleNamespace(mj_model=model))
  representation = SimpleNamespace(
    num_joints=1,
    extract_aux_state=lambda full_state: full_state[:, :7],
    represent=lambda full_state: full_state[:, 7:],
    joint_state=lambda rep_state: rep_state,
    compute_actor_sensor=lambda rep_state, sim_ctx, **kw: torch.cat(
      [rep_state[:, :1], kw["last_action"]], dim=1
    ),
    reconstruct_full_state=lambda aux_state, rep_state: torch.cat(
      [aux_state, rep_state], dim=1
    ),
  )
  contexts, used_actions, actor_inputs = [], [], []

  def dynamics(rep_state, action, aux_state, sim_ctx):
    used_actions.append(action.clone())
    torch.testing.assert_close(sim_ctx, torch.tensor([[0.5]]))
    predicted_aux = aux_state + torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    predicted_rep = rep_state + torch.cat([action, torch.zeros_like(action)], dim=1)
    return predicted_aux, predicted_rep

  def read_context(env):
    contexts.append(env)
    return torch.tensor([[0.5]])

  def actor(obs):
    actor_inputs.append(obs["actor"].clone())
    return torch.tensor([[0.5]])

  wm = SimpleNamespace(
    representation=representation,
    dynamics=dynamics,
    sim_context=SimpleNamespace(read=read_context),
    num_actions=1,
  )
  ghost = VelocityWorldModelGhost(wm, env, actor, rollout_steps=2)
  real_state = torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 4.0, 0.0]])
  history = torch.tensor([[[1.0, 0.0], [2.0, 1.0]]])
  real_action = torch.tensor([[0.25]])
  ghost.step(real_state, torch.tensor([[1.0, -1.0, 3.0]]), history, real_action)
  assert ghost.qpos == pytest.approx([0.0, 11.5, 22.0, 33.0, 1.0, 0.0, 0.0, 0.0, 4.25])

  # A changing real state only affects the next rollout seed.
  next_real_state = real_state.clone()
  next_real_state[:, 0] = 8.0
  ghost.step(next_real_state, torch.tensor([[-1.0, 1.0, -3.0]]), history, real_action)
  assert ghost.qpos == pytest.approx([0.0, 12.0, 22.0, 33.0, 1.0, 0.0, 0.0, 0.0, 4.75])
  torch.testing.assert_close(
    actor_inputs[-1], torch.tensor([[2.0, 1.0, 4.25, 0.25, -1.0, 1.0, -3.0]])
  )
  torch.testing.assert_close(torch.cat(used_actions), torch.tensor([[0.25], [0.5]]))

  ghost.step(next_real_state, torch.zeros(1, 3), history, real_action)
  assert ghost.qpos == pytest.approx([0.0, 18.5, 22.0, 33.0, 1.0, 0.0, 0.0, 0.0, 4.25])
  assert contexts == [env, env]
  assert model.qpos0[1] == 0.0
  assert next_real_state[0, 0] == 8.0

  drawn = []
  viz = SimpleNamespace(
    add_ghost_entity=lambda qpos, **kwargs: drawn.append(qpos.copy())
  )
  ghost.draw(viz)
  assert drawn[0] == pytest.approx(ghost.qpos)
  ghost.reset()
  ghost.draw(viz)
  assert ghost.qpos is None and len(drawn) == 1
