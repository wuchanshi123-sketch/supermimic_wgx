"""Regression coverage for the standalone G1 velocity task."""

import math
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from mjlab.managers import RewardManager
from mjlab.utils.lab_api.math import quat_apply_inverse
from tensordict import TensorDict

from supermimic.envs import velocity_world_model_env as velocity_imagination
from supermimic.envs.velocity_world_model_env import VelocityWorldModelEnv
from supermimic.replay_buffer import ReplayBuffer
from supermimic.representations import create_representation
from supermimic.runners.supermimic_runner import BaseRunner
from supermimic.runners.velocity_runner import VelocityRunner
from supermimic.scripts import train
from supermimic.tasks import velocity_env_cfg as velocity_task
from supermimic.tasks.env_cfgs import gencoord_env_cfg, maxcoord_env_cfg
from supermimic.tasks.velocity_env_cfg import (
  VelocityCommandCfg,
  advance_velocity_commands,
  velocity_env_cfg,
)
from supermimic.utils.gencoord_helpers import decompose_state

NUM_JOINTS = 29
NUM_ENVS = 4
STATE_DIM = 13 + 2 * NUM_JOINTS
DT = 0.02


def _representation():
  return create_representation(
    "gencoord",
    STATE_DIM,
    NUM_JOINTS,
    DT,
    4,
    torch.zeros(NUM_JOINTS),
    actor_obs_mode="proprio",
  )


def _state() -> torch.Tensor:
  # The first two rows face +world-Y, so forward motion is world-Y.
  half_sqrt_two = math.sqrt(0.5)
  state = torch.zeros(NUM_ENVS, STATE_DIM)
  state[:, 2] = torch.tensor([0.8, 0.8, 0.8, 0.2])
  state[:2, 3:7] = torch.tensor([half_sqrt_two, 0.0, 0.0, half_sqrt_two])
  state[2, 3:7] = torch.tensor([half_sqrt_two, half_sqrt_two, 0.0, 0.0])
  state[3, 3] = 1.0
  state[:2, 7 + NUM_JOINTS + 1] = 0.2
  return state


class _ForwardVelocityDynamics:
  def __call__(self, rep_state, action, aux_state, sim_ctx):
    del sim_ctx
    next_rep_state = rep_state.clone()
    next_rep_state[:, 4] += action[:, 0]
    return aux_state, next_rep_state


class _Scene(dict):
  env_origins = torch.zeros(NUM_ENVS, 3)


def test_velocity_config_keeps_tracking_configs_separate():
  velocity = velocity_env_cfg()
  tracking = gencoord_env_cfg()
  maxcoord = maxcoord_env_cfg()

  assert tuple(velocity.commands) == ("twist",)
  command = velocity.commands["twist"]
  assert isinstance(command, VelocityCommandCfg)
  assert command.lin_vel_x == (-1.0, 1.0)
  assert command.lin_vel_y == (-1.0, 1.0)
  assert command.ang_vel_z == (-3.0, 3.0)
  assert tuple(velocity.observations["actor"].terms) == tuple(
    tracking.observations["actor"].terms
  )
  assert tuple(velocity.observations["full_state"].terms) == tuple(
    tracking.observations["full_state"].terms
  )
  assert {"reset_base", "reset_robot_joints"} <= velocity.events.keys()
  assert {sensor.name for sensor in velocity.scene.sensors} == {
    "self_collision",
    "feet_ground_contact",
  }
  assert "motion_end" not in velocity.terminations
  assert "command_vel" not in velocity.curriculum
  for cfg in (tracking, maxcoord):
    assert tuple(cfg.commands) == ("motion",)
    assert "motion_end" in cfg.terminations
    assert "reset_robot_joints" not in cfg.events


@pytest.mark.parametrize(
  ("task_id", "standalone"),
  [
    ("SuperMimic-GenCoord-G1", False),
    ("SuperMimic-MaxCoord-G1", False),
    ("SuperMimic-GenCoord-G1-Velocity", True),
  ],
)
def test_train_routes_velocity_separately_from_motion_tasks(
  monkeypatch, task_id, standalone
):
  argv = ["train.py", task_id, "--num-envs", "4", "--runner.max-iterations", "2"]
  if not standalone:
    argv += ["--motion-file", "reference.npz"]
  monkeypatch.setattr(sys, "argv", argv)
  monkeypatch.setattr(train, "setup_device", lambda _device: "cpu")
  executed = []

  def unexpected_download(_name):
    pytest.fail("the training command attempted to download a reference motion")

  def create_env(cfg, num_envs, device):
    assert num_envs == 4 and device == "cpu"
    return cfg

  def make_runner(expected_standalone):
    def runner(env, cfg, **kwargs):
      assert standalone == expected_standalone
      assert cfg.max_iterations == 2
      assert kwargs["task_id"] == task_id
      if expected_standalone:
        assert tuple(env.commands) == ("twist",)
      else:
        assert tuple(env.commands) == ("motion",)
        assert env.commands["motion"].motion_file == "reference.npz"
        assert env.commands["motion"].sampling_mode == "external"
        assert kwargs["registry_name"] is None
      return SimpleNamespace(run=lambda: executed.append(task_id))

    return runner

  monkeypatch.setattr(train, "create_env", create_env)
  monkeypatch.setattr(train, "download_registry_motions", unexpected_download)
  monkeypatch.setattr(train, "VelocityRunner", make_runner(True))
  monkeypatch.setattr(train, "SuperMimicRunner", make_runner(False))
  train.main()
  assert executed == [task_id]


@pytest.mark.parametrize(("history_length", "scale_by_dt"), [(1, True), (3, False)])
@pytest.mark.parametrize("amp", [False, True])
def test_imagined_step_uses_pre_switch_command_and_matches_simulator_reward(
  monkeypatch, history_length, scale_by_dt, amp
):
  representation = _representation()
  cfg = velocity_env_cfg(amp=amp)
  cfg.scale_rewards_by_dt = scale_by_dt
  cfg.rewards["velocity"].weight = 1.25
  cfg.commands["twist"] = replace(
    cfg.commands["twist"], standing_fraction=1.0, single_axis_fraction=0.0
  )

  # The real MJCF-derived torso kinematics have their own validation. Keep this
  # test focused on the reward manager and imagined-step input contract.
  def torso_state(rep_state, root_quat_w):
    return 2.0 * root_quat_w[..., 1:3].square().sum(-1), rep_state[..., 4].square()

  joint_acc = torch.linspace(10.0, 50.0, NUM_JOINTS).expand(NUM_ENVS, NUM_JOINTS)
  collision_count = torch.tensor([1.0, 1.0, 0.0, 2.0])
  foot_slip = torch.tensor([0.4, 0.4, 0.1, 2.0])
  soft_landing = torch.tensor([150.0, 150.0, 50.0, 400.0])
  reward_signals = torch.stack(
    (joint_acc.square().sum(-1), collision_count, foot_slip, soft_landing), dim=-1
  )
  monkeypatch.setattr(
    velocity_imagination, "VelocityTorsoKinematics", lambda *_: torso_state
  )
  monkeypatch.setattr(
    velocity_imagination,
    "VelocityAngularMomentum",
    lambda _: lambda rep_state, _com: rep_state[..., 4:7].square().sum(-1),
  )
  monkeypatch.setattr(velocity_task, "self_collision_count", lambda _: collision_count)
  monkeypatch.setattr(
    velocity_task, "VelocityFootCosts", lambda _: lambda: (foot_slip, soft_landing)
  )
  limits = torch.tensor([-1.0, 1.0]).expand(NUM_ENVS, NUM_JOINTS, 2)
  joint_names = ["left_hip_pitch_joint", *(f"wrist_{i}" for i in range(1, NUM_JOINTS))]
  robot = SimpleNamespace(
    joint_names=joint_names,
    body_names=["torso_link"],
    data=SimpleNamespace(
      soft_joint_pos_limits=limits,
      default_joint_pos=torch.full((NUM_ENVS, NUM_JOINTS), 0.1),
    ),
  )
  scene = _Scene(robot=robot)
  sim = SimpleNamespace(
    cfg=cfg, scene=scene, step_dt=DT, device="cpu", num_envs=NUM_ENVS
  )
  sim.unwrapped = sim
  sim.num_actions = NUM_JOINTS
  sim.reward_manager = RewardManager(cfg.rewards, sim, scale_by_dt=scale_by_dt)
  simulator = VelocityWorldModelEnv(
    dynamics=_ForwardVelocityDynamics(),
    representation=representation,
    env=sim,
    batch_size=NUM_ENVS,
    replay_buffer=None,
    history_length=history_length,
    termination_softness=0.05,
    sim_context=SimpleNamespace(find=lambda _: None),
    reward_signals=lambda *_: reward_signals,
  )
  command = torch.tensor([0.55, 0.0, 0.0]).repeat(NUM_ENVS, 1)
  command[-1] = 0.0
  time_left = torch.tensor([DT / 2, 1.0, 1.0, 1.0])
  previous = torch.full((NUM_ENVS, NUM_JOINTS), 0.05)
  batch = TensorDict(
    {
      "state": _state(),
      "prev_action": previous,
      "sim_ctx": torch.zeros(NUM_ENVS, 2),
      "velocity_command": command.clone(),
      "command_time_left": time_left.clone(),
      "sensor_window": torch.zeros(
        NUM_ENVS, history_length, representation.proprio_obs_dim(NUM_JOINTS)
      ),
      "pad_mask": torch.zeros(NUM_ENVS, history_length, dtype=torch.bool),
    },
    batch_size=[NUM_ENVS],
  )
  initial_obs = simulator.reset(batch)
  torch.testing.assert_close(initial_obs["actor"][:, -3:], command)
  assert initial_obs["actor"].shape == (
    NUM_ENVS,
    history_length * representation.proprio_obs_dim(NUM_JOINTS) + 3,
  )
  assert initial_obs["critic"].shape == (NUM_ENVS, representation.rep_state_dim + 5)

  action = torch.full((NUM_ENVS, NUM_JOINTS), 0.1, requires_grad=True)
  obs, predicted_reward, term_prob, _ = simulator.step(action)
  # Identical post-step physics and the original command give identical rewards,
  # even when one command has expired and the other has not.
  torch.testing.assert_close(predicted_reward[0], predicted_reward[1])
  torch.testing.assert_close(obs["actor"][1:, -3:], command[1:])
  assert (
    simulator.command_time_left[0] >= cfg.commands["twist"].resampling_time_range[0]
  )
  torch.testing.assert_close(simulator.command_time_left[1:], time_left[1:] - DT)
  torch.testing.assert_close(obs["actor"][0, -3:], torch.zeros(3))
  torch.testing.assert_close(batch["velocity_command"], command)

  full_state = representation.reconstruct_full_state(
    simulator.aux_state, simulator.rep_state
  ).detach()
  root_pos, root_quat, joint_pos, lin_vel_w, ang_vel_w, joint_vel = decompose_state(
    full_state
  )
  gravity_w = root_pos.new_tensor([0.0, 0.0, -1.0]).expand_as(root_pos)
  lin_vel_b = quat_apply_inverse(root_quat, lin_vel_w)
  torso_omega = torch.zeros_like(lin_vel_b)
  torso_omega[..., 0] = lin_vel_b[..., 0]
  robot.data = SimpleNamespace(
    root_link_pos_w=root_pos,
    root_link_quat_w=root_quat,
    root_link_lin_vel_w=lin_vel_w,
    root_link_ang_vel_w=ang_vel_w,
    root_link_lin_vel_b=lin_vel_b,
    root_link_ang_vel_b=quat_apply_inverse(root_quat, ang_vel_w),
    projected_gravity_b=quat_apply_inverse(root_quat, gravity_w),
    body_link_quat_w=root_quat.unsqueeze(-2),
    body_link_ang_vel_w=torso_omega.unsqueeze(-2),
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    joint_acc=joint_acc,
    soft_joint_pos_limits=limits,
  )
  if not amp:
    scene["robot/root_angmom"] = SimpleNamespace(data=lin_vel_b)
  torch.testing.assert_close(lin_vel_w[:2, 1], torch.full((2,), 0.3))
  sim.command_manager = SimpleNamespace(
    get_term=lambda _: SimpleNamespace(command=command)
  )
  sim.action_manager = SimpleNamespace(action=action.detach(), prev_action=previous)
  fall = cfg.terminations["fall"]
  actual_fall = fall.func(sim, **fall.params)
  torch.testing.assert_close(actual_fall, torch.tensor([False, False, True, True]))
  torch.testing.assert_close(term_prob >= 0.5, actual_fall)
  sim.termination_manager = SimpleNamespace(terminated=actual_fall)
  actual_reward = sim.reward_manager.compute(dt=DT)
  torch.testing.assert_close(
    sim.reward_manager.get_term_cfg("velocity").func.signals, reward_signals
  )
  term = cfg.rewards["velocity"]
  fall_cost = (
    term.params["cfg"].terminated_cost * term.weight * (DT if scale_by_dt else 1.0)
  )
  # The same cost is charged on a hard fall in MuJoCo and on the expected fall
  # indicator in the differentiable rollout.
  expected_reward = actual_reward + fall_cost * (actual_fall.float() - term_prob)
  torch.testing.assert_close(predicted_reward, expected_reward)

  predicted_reward.sum().backward()
  assert action.grad is not None and torch.isfinite(action.grad).all()
  assert action.grad[1, 0] > 0.0


@pytest.mark.parametrize("single_axis_fraction", [0.0, 1.0])
def test_sampler_covers_positive_and_negative_motion(single_axis_fraction):
  sample_count = 256
  cfg = VelocityCommandCfg(
    standing_fraction=0.0, single_axis_fraction=single_axis_fraction
  )
  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(7)
    sampled, _ = advance_velocity_commands(
      torch.zeros(sample_count, 3), torch.zeros(sample_count), dt=DT, cfg=cfg
    )
  expected_axes = 1 if single_axis_fraction else 3
  assert torch.all((sampled != 0.0).sum(-1) == expected_axes)
  for axis, (lower, upper) in enumerate((cfg.lin_vel_x, cfg.lin_vel_y, cfg.ang_vel_z)):
    assert lower <= sampled[:, axis].min() < 0.0
    assert 0.0 < sampled[:, axis].max() <= upper


def test_velocity_replay_snapshots_command_before_step_and_keeps_continuous_windows():
  runner = VelocityRunner.__new__(VelocityRunner)
  runner._representation = _representation()
  runner._num_actions = NUM_JOINTS
  runner._sim_context = SimpleNamespace(output_dim=2)
  runner.cfg = SimpleNamespace(history_length=1)
  runner.device = "cpu"
  command = torch.ones(NUM_ENVS, 3)
  countdown = torch.full((NUM_ENVS,), DT)
  runner.velocity_command = SimpleNamespace(command=command, time_left=countdown)
  actions = torch.zeros(NUM_ENVS, NUM_JOINTS)
  runner.action_manager = SimpleNamespace(action=actions)
  runner._sim_reward = SimpleNamespace(signals=torch.zeros(NUM_ENVS, 4))
  first_step_signals = torch.tensor([10.0, 1.0, 2.0, 300.0])

  def step(_actions):
    command.add_(1)
    countdown.add_(1)
    runner._sim_reward.signals.add_(first_step_signals)
    return {}, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS, dtype=torch.bool), {}

  runner.env = SimpleNamespace(num_envs=NUM_ENVS, step=step)
  schema = runner._buffer_schema()
  assert "motion_time" in BaseRunner._task_buffer_schema(runner)
  assert "motion_time" not in schema
  runner.replay_buffer = ReplayBuffer(schema, 8, NUM_ENVS, "cpu")
  for _ in range(2):
    runner._step_and_store(
      actions, _state(), extra_fields={"sim_ctx": torch.zeros(NUM_ENVS, 2)}
    )
  frames = runner.replay_buffer.sample(sequence_length=2, batch_size=2)
  torch.testing.assert_close(
    frames["velocity_command"], torch.tensor([1.0, 2.0]).view(1, 2, 1).expand(2, 2, 3)
  )
  torch.testing.assert_close(
    frames["command_time_left"], torch.tensor([DT, 1 + DT]).expand(2, 2)
  )
  torch.testing.assert_close(
    frames["velocity_reward_signals"],
    torch.stack((first_step_signals, 2.0 * first_step_signals)).expand(2, 2, 4),
  )
