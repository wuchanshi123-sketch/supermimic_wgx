"""Regression tests for the optional velocity-task AMP integration."""

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torch import nn

from supermimic.envs import velocity_world_model_env as velocity_imagination
from supermimic.envs.velocity_world_model_env import VelocityWorldModelEnv
from supermimic.learners.amp_style import AmpStyleCfg
from supermimic.learners.velocity_wm_learner import (
  VelocityAmpTraining,
  VelocityWorldModelLearner,
)
from supermimic.learners.wm_learner import WorldModelLearner
from supermimic.representations import create_representation
from supermimic.runners.velocity_runner import (
  AMP_DISCRIMINATOR_VERSION,
  VELOCITY_AMP_TASK_ID,
  VELOCITY_TASK_ID,
  VelocityRunner,
  VelocityRunnerCfg,
)
from supermimic.scripts import play, train
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.velocity_env_cfg import velocity_env_cfg
from supermimic.utils.gencoord_helpers import full_state_offsets

NUM_JOINTS = 29
NUM_ENVS = 4
DT = 0.02


class _JointDynamics:
  def __call__(self, rep_state, action, aux_state, sim_ctx):
    del sim_ctx
    next_rep_state = rep_state.clone()
    next_rep_state[..., 10] += action[..., 0]
    next_rep_state[..., 4:6] += action[..., 1:3]
    next_rep_state[..., 9] += action[..., 3]
    next_rep_state[..., 6:8] += action[..., 4:6]
    return aux_state, next_rep_state


class _StyleReward(nn.Module):
  def reward(self, current_rep, next_rep):
    return (next_rep[..., 10] - current_rep[..., 10]).clamp(0.0, 1.0)


class _BodyFeatures:
  def __init__(self):
    self.kinematics = SimpleNamespace(
      body_kinematics=lambda rep: rep,
      from_kinematics=lambda rep, _com: torch.zeros_like(rep[:, 0]),
    )

  def __call__(self, rep, _kinematics=None):
    return rep


@pytest.mark.parametrize("scale_by_dt", [True, False])
def test_imagined_amp_reward_uses_current_command_and_next_state_speed(
  monkeypatch, scale_by_dt
):
  cfg = velocity_env_cfg()
  cfg.scale_rewards_by_dt = scale_by_dt
  cfg.rewards["velocity"].weight = 0.5
  cfg.commands["twist"] = replace(
    cfg.commands["twist"], standing_fraction=1.0, single_axis_fraction=0.0
  )
  monkeypatch.setattr(
    velocity_imagination,
    "VelocityTorsoKinematics",
    lambda _: lambda rep, _quat: (torch.zeros_like(rep[:, 0]),) * 2,
  )
  monkeypatch.setattr(
    velocity_imagination,
    "VelocityAngularMomentum",
    lambda _: lambda rep, _com: torch.zeros_like(rep[:, 0]),
  )
  task_reward = cfg.rewards["velocity"]
  term = replace(
    task_reward,
    func=SimpleNamespace(
      evaluate=lambda rep, *args, **kwargs: torch.full_like(rep[:, 0], 0.4)
    ),
  )
  limits = torch.tensor([-1.0, 1.0]).expand(1, NUM_JOINTS, 2)
  robot = SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=limits))
  env = SimpleNamespace(
    cfg=cfg,
    num_actions=NUM_JOINTS,
    step_dt=DT,
    reward_manager=SimpleNamespace(get_term_cfg=lambda _: term),
    scene={"robot": robot},
  )
  env.unwrapped = env
  state_dim = 13 + 2 * NUM_JOINTS
  representation = create_representation(
    "gencoord",
    state_dim,
    NUM_JOINTS,
    DT,
    4,
    torch.zeros(NUM_JOINTS),
    actor_obs_mode="proprio",
  )
  env_kwargs = {
    "dynamics": _JointDynamics(),
    "representation": representation,
    "env": env,
    "batch_size": NUM_ENVS,
    "replay_buffer": None,
    "history_length": 1,
    "termination_softness": 0.05,
    "sim_context": SimpleNamespace(find=lambda _: None),
    "reward_signals": lambda *args: torch.zeros(NUM_ENVS, 4),
  }
  baseline = VelocityWorldModelEnv(**env_kwargs)
  style_cfg = AmpStyleCfg(
    motion_file="reference.npz", weight=0.8, style_command_gate=0.2
  )
  amp = VelocityWorldModelEnv(
    **env_kwargs,
    amp_discriminator=_StyleReward(),
    amp_cfg=style_cfg,
    amp_features=_BodyFeatures(),
  )

  full_state = torch.zeros(NUM_ENVS, state_dim)
  full_state[:, 2] = 0.76
  full_state[:, 3] = 1.0
  command = torch.tensor(
    [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, -0.5], [0.0, 0.1, 0.0]]
  )
  _, root_lin, root_ang, _ = full_state_offsets(NUM_JOINTS)
  full_state[:, root_lin : root_lin + 2] = command[:, :2]
  full_state[:, root_ang + 2] = command[:, 2]
  initial = TensorDict(
    {
      "state": full_state,
      "prev_action": torch.zeros(NUM_ENVS, NUM_JOINTS),
      "sim_ctx": torch.zeros(NUM_ENVS, 4),
      "velocity_command": command,
      "command_time_left": torch.tensor([1.0, DT * 0.5, 1.0, 1.0]),
    },
    [NUM_ENVS],
  )
  baseline_obs = baseline.reset(initial)
  amp_obs = amp.reset(initial)
  assert baseline_obs["actor"].shape == amp_obs["actor"].shape
  assert baseline_obs["critic"].shape == amp_obs["critic"].shape
  action = torch.zeros(NUM_ENVS, NUM_JOINTS)
  action[:, 0] = 0.25
  action[1, 1:4] = torch.tensor([0.25, -0.15, -0.2])
  action[2, 3] = 0.5
  action[:, 4:6] = 0.25
  action.requires_grad_()
  _, baseline_reward, _, _ = baseline.step(action)
  obs, amp_reward, _, _ = amp.step(action)

  dt_scale = DT if scale_by_dt else 1.0
  expected_scale = style_cfg.weight * dt_scale
  gate = torch.tensor([0.0, 1.0, 1.0, 0.5])
  task_cfg = task_reward.params["cfg"]
  speed_std = torch.tensor([task_cfg.linear_std, task_cfg.linear_std, task_cfg.yaw_std])
  speed_offset = action.detach()[:, 1:4]
  fidelity = (-(speed_offset / speed_std).square().sum(-1)).exp()
  style_bonus = 0.25 * expected_scale * gate * fidelity
  torch.testing.assert_close(baseline_reward, torch.full((NUM_ENVS,), 0.2 * dt_scale))
  torch.testing.assert_close(amp_reward - baseline_reward, style_bonus)
  torch.testing.assert_close(obs["actor"][1, -3:], torch.zeros(3))
  assert amp.max_step_reward == pytest.approx(baseline.max_step_reward + expected_scale)
  gradient = torch.autograd.grad(amp_reward.sum(), action)[0]
  torch.testing.assert_close(gradient[:, 0], gate * expected_scale * fidelity)
  torch.testing.assert_close(
    gradient[:, 1:4], -2.0 * style_bonus[:, None] * speed_offset / speed_std.square()
  )
  torch.testing.assert_close(gradient[:, 4:], torch.zeros_like(gradient[:, 4:]))


@pytest.mark.parametrize(
  ("task_id", "motion_file"),
  [
    (VELOCITY_TASK_ID, None),
    (VELOCITY_TASK_ID, "custom.npz"),
    (VELOCITY_AMP_TASK_ID, None),
  ],
)
def test_velocity_train_routes_optional_amp_config(monkeypatch, task_id, motion_file):
  argv = ["train.py", task_id, "--num-envs", "4"]
  if motion_file is not None:
    argv += ["--runner.amp.motion-file", motion_file, "--runner.amp.weight", "0.75"]
  monkeypatch.setattr(sys, "argv", argv)
  monkeypatch.setattr(train, "setup_device", lambda _: "cpu")
  monkeypatch.setattr(train, "create_env", lambda cfg, n, device: cfg)
  executed = []

  def make_runner(env, cfg, **kwargs):
    assert tuple(env.commands) == ("twist",)
    if task_id == VELOCITY_AMP_TASK_ID:
      dataset = Path(__file__).resolve().parents[2] / "WalkandRun_amp_walk_50hz"
      assert cfg.amp.motion_file == str(dataset)
    else:
      assert cfg.amp.motion_file == motion_file
    assert cfg.amp.weight == (0.75 if motion_file else 1.25)
    return SimpleNamespace(run=lambda: executed.append(kwargs["task_id"]))

  monkeypatch.setattr(train, "VelocityRunner", make_runner)
  train.main()
  assert executed == [task_id]


def test_velocity_amp_play_routes_to_existing_gamepad_player(monkeypatch):
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "play.py",
      VELOCITY_AMP_TASK_ID,
      "--checkpoint",
      "actor.pt",
      "--gamepad-device",
      "/dev/input/js1",
    ],
  )
  received = []
  monkeypatch.setattr(play, "play_velocity", received.append)
  play.main()
  (cfg,) = received
  assert cfg.checkpoint == Path("actor.pt")
  assert cfg.gamepad_device == Path("/dev/input/js1")


def test_amp_discriminator_uses_adjacent_real_frames_at_all_commands(monkeypatch):
  cfg = AmpStyleCfg(batch_size=32, num_batches=2)
  received = []
  expert_features = torch.tensor([[-3.0], [-2.0], [-1.0], [8.0], [9.0]])
  starts = torch.tensor([0, 3]).repeat(cfg.batch_size // 2)
  expected_expert = torch.stack(
    (expert_features[starts], expert_features[starts + 1]), dim=1
  )

  def sample_indices(batch_size):
    assert batch_size == cfg.batch_size
    return starts

  def update_style(expert, policy):
    assert not expert.requires_grad and not policy.requires_grad
    torch.testing.assert_close(expert, expected_expert)
    received.append(policy)
    return {"AMP/loss": torch.tensor(float(len(received)))}

  learner = VelocityWorldModelLearner.__new__(VelocityWorldModelLearner)
  learner.representation = SimpleNamespace(represent=lambda states: states)
  learner._reward_signals = nn.Linear(1, 1)
  learner._fit_signals = lambda _: {}
  learner._amp = VelocityAmpTraining(
    SimpleNamespace(cfg=cfg, update=update_style),
    SimpleNamespace(sample_indices=sample_indices),
    _BodyFeatures(),
    expert_features.requires_grad_(),
  )
  monkeypatch.setattr(WorldModelLearner, "update", lambda *args: {})

  commands = torch.zeros(2, 3, 3)
  commands[0, 1:, 0] = 0.5
  commands[1, 0, 2] = -0.7
  states = torch.arange(6.0).reshape(2, 3, 1).requires_grad_()
  window = TensorDict(
    {
      "state": states,
      "velocity_command": commands,
      "action": torch.zeros(2, 3, 1),
      "sim_ctx": torch.zeros(2, 3, 1),
      "velocity_reward_signals": torch.zeros(2, 3, 1),
    },
    [2, 3],
  )

  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(17)
    assert learner.update(iter([window]))["AMP/loss"].item() == 1.5
    window["velocity_command"].zero_()
    assert learner.update(iter([window]))["AMP/loss"].item() == 3.5

  assert len(received) == 2 * cfg.num_batches
  for policy in received:
    pairs = policy.squeeze(-1).unique(dim=0)
    torch.testing.assert_close(pairs, torch.tensor([[0.0, 1.0], [3.0, 4.0]]))


def test_velocity_amp_checkpoint_migrates_legacy_and_restores_versioned_head(tmp_path):
  cfg = VelocityRunnerCfg(
    actor_hidden_dims=(8,),
    critic_hidden_dims=(8,),
    dynamics_hidden_dims=(8,),
    buffer_size=8,
  )
  env = create_env(velocity_env_cfg(), num_envs=2, device="cpu")
  try:
    baseline = VelocityRunner(env, cfg, VELOCITY_TASK_ID, device="cpu")
    old_run = tmp_path / "baseline"
    old_run.mkdir()
    baseline._save_checkpoint(old_run / "checkpoint_0")
    old_bundle = torch.load(old_run / "world_model.pt", weights_only=True)
    assert "amp_discriminator" not in old_bundle

    motion = tmp_path / "walk.npz"
    joint_names = env.unwrapped.scene["robot"].joint_names
    zeros = np.zeros((3, len(joint_names)), dtype=np.float32)
    np.savez(
      motion,
      fps=50.0,
      joint_names=np.array(joint_names),
      joint_pos=zeros,
      joint_vel=zeros,
      root_pos_w=np.tile([0.0, 0.0, 0.76], (3, 1)),
      root_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
      root_lin_vel_w=np.zeros((3, 3)),
      root_ang_vel_w=np.zeros((3, 3)),
    )
    cfg_amp = replace(
      cfg,
      init_checkpoint=str(old_run),
      amp=AmpStyleCfg(motion_file=str(motion), hidden_dims=(8,)),
    )
    amp = VelocityRunner(env, cfg_amp, VELOCITY_AMP_TASK_ID, device="cpu")
    torch.testing.assert_close(
      baseline.policy_learner.actor.state_dict(), amp.policy_learner.actor.state_dict()
    )
    assert amp._amp is not None
    discriminator = amp._amp.learner.discriminator
    reference_mean = discriminator.feature_mean.clone()
    with torch.no_grad():
      discriminator.network[-1].bias.fill_(0.23)
      discriminator.feature_mean.add_(0.45)
    amp_run = tmp_path / "amp"
    amp_run.mkdir()
    amp._save_checkpoint(amp_run / "checkpoint_1")
    bundle = torch.load(amp_run / "world_model.pt", weights_only=True)
    assert bundle["amp_discriminator_version"] == AMP_DISCRIMINATOR_VERSION == 3
    resumed = VelocityRunner(
      env,
      replace(cfg_amp, init_checkpoint=str(amp_run)),
      VELOCITY_AMP_TASK_ID,
      device="cpu",
    )
    assert resumed._amp is not None
    torch.testing.assert_close(
      discriminator.state_dict(), resumed._amp.learner.discriminator.state_dict()
    )

    legacy_path = tmp_path / "legacy_amp"
    legacy_path.mkdir()
    bundle.pop("amp_discriminator_version")
    torch.save(bundle, legacy_path / "world_model.pt")
    torch.save(amp.policy_learner.actor.state_dict(), legacy_path / "actor.pt")
    migrated = VelocityRunner(
      env,
      replace(cfg_amp, init_checkpoint=str(legacy_path)),
      VELOCITY_AMP_TASK_ID,
      device="cpu",
    )
    assert migrated._amp is not None
    torch.testing.assert_close(
      migrated._amp.learner.discriminator.feature_mean, reference_mean, atol=0, rtol=0
    )
    torch.testing.assert_close(
      migrated.policy_learner.actor.state_dict(),
      amp.policy_learner.actor.state_dict(),
      atol=0,
      rtol=0,
    )
    torch.testing.assert_close(
      migrated._dynamics.state_dict(), amp._dynamics.state_dict(), atol=0, rtol=0
    )
  finally:
    env.close()
