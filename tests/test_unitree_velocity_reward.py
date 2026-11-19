"""Check Unitree's G1 reward equations and command boundaries."""

import math
from dataclasses import replace

import pytest
import torch

from supermimic.utils.unitree_velocity_reward import (
  UnitreeVelocityPosture,
  UnitreeVelocityRewardCfg,
  unitree_g1_home_joint_pos,
  unitree_velocity_reward,
)


def test_official_home_pose_and_posture_thresholds():
  names = (
    "robot/left_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "left_elbow_joint",
    "waist_yaw_joint",
  )
  target = unitree_g1_home_joint_pos(
    names, torch.zeros(len(names), dtype=torch.float64)
  )
  torch.testing.assert_close(target, target.new_tensor([0.35, -0.18, 0.87, 0.0]))
  posture = UnitreeVelocityPosture(names, target)

  # A pure yaw command also changes the posture profile. At 0.1 the walking
  # profile and the stand-still penalty both apply, as in the Unitree task.
  yaw = target.new_tensor([0.0, 0.1, 0.10001, 1.5])
  zero = torch.zeros_like(yaw)
  command = torch.stack((zero, zero, yaw), dim=-1)
  previous_action = target.new_zeros((len(yaw), len(target)))
  delta = target.new_tensor([0.1, 0.0, 0.1, 0.15])
  joint_pos = target + delta.expand_as(previous_action)
  state = torch.cat((target.new_zeros((len(yaw), 10)), joint_pos, previous_action), -1)
  limits = target.new_tensor([-5.0, 5.0]).expand(len(target), 2)
  cfg = replace(UnitreeVelocityRewardCfg(), linear_weight=0.0, yaw_weight=0.0)
  reward = unitree_velocity_reward(
    state,
    command,
    previous_action,
    previous_action,
    limits,
    cfg,
    posture=posture,
    torso_tilt_error=zero,
    torso_roll_pitch_speed_sq=zero,
    joint_acc_l2=zero,
    angular_momentum_l2=zero,
    self_collision_count=zero,
    terminated=zero,
    foot_slip=zero,
    soft_landing=zero,
  )

  expected_widths = target.new_tensor(
    [
      [0.05, 0.05, 0.05, 0.05],
      [0.15, 0.1, 0.1, 0.15],
      [0.15, 0.1, 0.1, 0.15],
      [0.25, 0.1, 0.1, 0.25],
    ]
  )
  expected_pose = torch.exp(-(delta / expected_widths).square().mean(-1))
  stand_still = (yaw <= 0.1) * delta.square().sum()
  torch.testing.assert_close(reward, expected_pose - stand_still)


def test_official_tracking_and_non_foot_costs():
  dtype = torch.float64
  target = torch.zeros(2, dtype=dtype)
  posture = UnitreeVelocityPosture(
    ("left_shoulder_pitch_joint", "left_knee_joint"), target
  )
  state = torch.tensor(
    [[0.8, 0, 0, -1, 0.3, -0.2, 0.1, 0.1, 0.2, 0.35, 0.75, 0, 0, 0]], dtype=dtype
  )
  command = torch.tensor([[0.2, 0.1, 0.4]], dtype=dtype)
  action = torch.tensor([[0.3, -0.7]], dtype=dtype)
  last_action = torch.tensor([[0.1, -0.2]], dtype=dtype)
  limits = torch.tensor([[-0.6, 0.6], [-1.0, 1.0]], dtype=dtype)
  terminated = torch.tensor([0.2], dtype=dtype, requires_grad=True)
  collision_count = torch.tensor([3.0], dtype=dtype, requires_grad=True)
  cfg = UnitreeVelocityRewardCfg()
  reward = unitree_velocity_reward(
    state,
    command,
    action,
    last_action,
    limits,
    cfg,
    posture=posture,
    torso_tilt_error=torch.tensor([0.3], dtype=dtype),
    torso_roll_pitch_speed_sq=torch.tensor([0.4], dtype=dtype),
    joint_acc_l2=torch.tensor([40000.0], dtype=dtype),
    angular_momentum_l2=torch.tensor([2.0], dtype=dtype),
    self_collision_count=collision_count,
    terminated=terminated,
    foot_slip=torch.zeros_like(terminated),
    soft_landing=torch.zeros_like(terminated),
  )
  expected = (
    math.exp(-(0.1**2 + 0.3**2 + 2 * 0.1**2) / 0.25)
    + math.exp(-(0.05**2 + 0.05 * (0.1**2 + 0.2**2)) / 0.5)
    + math.exp(-((0.75 / 0.15) ** 2) / 2)
    - (1 - 0.7**2)
    - 0.05 * 0.4
    - 0.025 * 2
    - 200 * 0.2
    - 2.5e-7 * 40000
    - 10 * 0.15
    - 0.05 * (0.2**2 + 0.5**2)
    - 3.0
  )
  assert reward.item() == pytest.approx(expected)
  assert cfg.max_reward == 3.0
  reward.sum().backward()
  assert terminated.grad.item() == -200.0
  assert collision_count.grad.item() == -1.0


def test_official_foot_costs_use_strict_planar_and_yaw_command_gate():
  commands = torch.tensor(
    [
      [0.0, 0.0, 0.0],
      [0.1, 0.0, 0.0],
      [0.0, 0.0, -0.1],
      [0.0, 0.10001, 0.0],
      [0.0, 0.0, -0.10001],
      [0.048, 0.064, 0.04],
    ],
    dtype=torch.float64,
  )
  zeros = commands.new_zeros(len(commands))
  actions = zeros.unsqueeze(-1)
  state = commands.new_zeros(len(commands), 12)
  posture = UnitreeVelocityPosture(("left_knee_joint",), commands.new_zeros(1))
  limits = commands.new_tensor([[-1.0, 1.0]])
  cfg = UnitreeVelocityRewardCfg()
  slip = torch.full_like(zeros, 2.0, requires_grad=True)
  landing = torch.full_like(zeros, 300.0, requires_grad=True)

  def score(foot_slip, soft_landing):
    return unitree_velocity_reward(
      state,
      commands,
      actions,
      actions,
      limits,
      cfg,
      posture=posture,
      torso_tilt_error=zeros,
      torso_roll_pitch_speed_sq=zeros,
      joint_acc_l2=zeros,
      angular_momentum_l2=zeros,
      self_collision_count=zeros,
      terminated=zeros,
      foot_slip=foot_slip,
      soft_landing=soft_landing,
    )

  reward = score(slip, landing)
  active = commands.new_tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
  torch.testing.assert_close(reward - score(zeros, zeros), -0.8 * active)
  reward.sum().backward()
  torch.testing.assert_close(slip.grad, -0.25 * active)
  torch.testing.assert_close(landing.grad, -0.001 * active)
