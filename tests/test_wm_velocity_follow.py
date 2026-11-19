"""Velocity following scores the applied command and separates task outcomes."""

import math
from dataclasses import replace
from types import SimpleNamespace

import torch

from supermimic.envs.multi_task_world_model_env import MultiTaskWorldModelEnv
from supermimic.tasks.task_spec import TaskSpec, VelocityTaskCfg
from supermimic.utils.velocity_reward import (
  heading_velocity,
  velocity_fall_probability,
  velocity_fallen,
)
from tests.conftest import (
  ACTION_DIM,
  MOTION_GOAL_DIM,
  identity_aux,
  make_mock_buffer,
  make_mock_command_env,
  make_mock_dynamics,
  make_mock_wm_representation,
)


def test_heading_velocity_ignores_root_tilt_and_uses_world_yaw_rate():
  # Pitch 45°, then yaw 90°. Local and world z axes differ.
  c = math.cos(math.pi / 8) * math.sqrt(0.5)
  s = math.sin(math.pi / 8) * math.sqrt(0.5)
  root_quat = torch.tensor([[c, -s, s, c]])
  lin_w = torch.tensor([[0.0, 0.8, 0.5]])
  ang_w = torch.tensor([[0.5, 0.0, 0.3]])
  torch.testing.assert_close(
    heading_velocity(root_quat, lin_w, ang_w), torch.tensor([[0.8, 0.0, 0.3]])
  )


def test_velocity_fall_uses_absolute_height_and_tilt():
  aux = identity_aux(3)
  aux[:, 2] = torch.tensor([0.85, 0.30, 0.85])
  aux[2, 3:7] = torch.tensor([math.sqrt(0.5), math.sqrt(0.5), 0, 0])
  fall = torch.tensor([False, True, True])
  assert torch.equal(velocity_fallen(aux[:, :3], aux[:, 3:7]), fall)
  assert torch.equal(
    velocity_fall_probability(aux[:, :3], aux[:, 3:7], 0.03) > 0.5, fall
  )


def test_velocity_reward_uses_inherited_command_before_resampling():
  batch_size, velocity_start = 4, 2
  rep = make_mock_wm_representation(batch_size)
  rep.compute_wm_reward_from_reference.side_effect = lambda rep_state, *args: (
    torch.full((len(rep_state),), 0.1)
  )
  body_rot = torch.eye(3).expand(2, 2, 3, 3)
  body_velocity = torch.zeros(2, 2, 3)
  rep.world_frame_bodies.return_value = (
    body_velocity,
    body_rot,
    body_velocity,
    body_velocity,
  )
  sim = make_mock_command_env()
  sim.num_envs = 8
  sim.unwrapped.step_dt = 0.02
  sim.unwrapped.command_manager.get_term.return_value.cfg.body_names = (
    "pelvis",
    "torso_link",
  )
  limits = torch.tensor([-10.0, 10.0]).expand(8, ACTION_DIM, 2)
  sim.unwrapped.scene = {
    "robot": SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=limits))
  }
  dyn = make_mock_dynamics()
  next_aux = identity_aux(batch_size)
  next_aux[:, 2] = 0.8
  dyn.side_effect = lambda state, *args: (next_aux, torch.zeros_like(state))
  replay = make_mock_buffer()
  motion = TaskSpec(
    name="motion", goal_dim=MOTION_GOAL_DIM, extract_training_goal=lambda *args: None
  )
  velocity = replace(
    motion,
    name="velocity",
    goal_dim=3,
    precompute_replay_goal=lambda _: torch.tensor([[0.5, 0.0, 0.0]]).expand(10, 3),
  )
  env = MultiTaskWorldModelEnv(
    dynamics=dyn,
    representation=rep,
    env=sim,
    batch_size=batch_size,
    replay_buffer=replay,
    task_specs=[motion, velocity],
    history_length=1,
    termination_softness=0.02,
    velocity_cfg=VelocityTaskCfg(mode="follow", reference_probability=1.0),
    velocity_start=velocity_start,
    sim_velocity_start=4,
  )
  batch = replay.sample(1, batch_size).squeeze(1)
  batch["motion_time"] = torch.tensor([0, 0, 9, 7])
  batch["prev_action"] = torch.zeros(batch_size, ACTION_DIM)
  batch["velocity_command"] = torch.zeros(batch_size, 3)
  batch["velocity_reference"] = torch.zeros(batch_size, dtype=torch.bool)
  batch["velocity_steps"] = torch.ones(batch_size, dtype=torch.long)

  obs = env.reset(batch)
  assert torch.equal(obs["goal_velocity"], torch.zeros(batch_size, 3))
  assert torch.equal(obs["critic"][:, -1], torch.tensor([0.0, 0.0, 1.0, 1.0]))
  obs, reward, term, _ = env.step(torch.zeros(batch_size, ACTION_DIM))
  # Zero measured speed matches the inherited zero command for that action.
  torch.testing.assert_close(reward, torch.tensor([0.1, 0.1, 0.08, 0.08]))
  assert (term[:velocity_start] > 0.9).all()  # Far above reference height.
  assert (term[velocity_start:] < 1e-6).all()  # Upright and above the fall threshold.
  # Both velocity slots select reference commands. The frozen end pose in the
  # first slot contributes no imitation, but the command still follows it.
  assert torch.equal(
    obs["velocity_reference"], torch.tensor([False, False, False, True])
  )
  torch.testing.assert_close(
    obs["goal_velocity"][velocity_start:, 0], torch.full((2,), 0.5)
  )
