from dataclasses import replace
from types import SimpleNamespace

import torch
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from supermimic.tasks.task_spec import VelocityTaskCfg, make_velocity_task
from supermimic.tasks.velocity_commands import VelocityCommands, reference_bc_mask


def test_moving_commands_cover_all_bins_in_a_full_renewal():
  cfg = VelocityTaskCfg(
    mode="follow",
    command_bins=(2, 3, 2),
    reference_probability=0.0,
    standing_probability=0.0,
    lin_vel_x_range=(-1.0, 1.0),
    lin_vel_y_range=(-1.5, 1.5),
    ang_vel_z_range=(-1.0, 1.0),
  )
  commands = VelocityCommands(cfg, torch.zeros(2, 3), 12, 0.02, "cpu")
  fixed, source, remaining = commands.sample()
  cells = torch.floor(fixed + torch.tensor([1.0, 1.5, 1.0])).long()
  expected = torch.cartesian_prod(torch.arange(2), torch.arange(3), torch.arange(2))
  assert torch.equal(cells.unique(dim=0), expected)
  assert not source.any()
  assert (remaining == 250).all()


def test_command_hold_and_reference_source_behavior():
  reference = torch.tensor([[0.1, 0.0, 0.0], [0.6, 0.0, 0.0]])
  cfg = VelocityTaskCfg(mode="follow", hold_seconds=0.04, reference_probability=0.0)
  sampler = VelocityCommands(cfg, reference, 3, 0.02, "cpu")
  fixed, source, _ = sampler.sample()
  remaining = torch.tensor([1, 2, 2])
  next_fixed, _, next_remaining = sampler.advance(
    fixed, source, remaining, reset=torch.tensor([0, 0, 1])
  )
  assert torch.equal(next_fixed[1], fixed[1])
  assert torch.equal(next_remaining, torch.tensor([2, 1, 2]))

  cfg = replace(cfg, reference_probability=1.0)
  sampler = VelocityCommands(cfg, reference, 3, 0.02, "cpu")
  fixed, source, _ = sampler.sample()
  assert source.all()
  assert torch.equal(
    sampler.goal(fixed, source, torch.tensor([0, 1, 0])), reference[[0, 1, 0]]
  )


def test_reference_bc_is_disabled_at_each_clip_end():
  ends = torch.tensor([3, 7])
  frames = torch.tensor([1, 2, 3, 4, 6])
  source = torch.tensor([True, True, True, False, True])
  assert torch.equal(
    reference_bc_mask(source, frames, ends),
    torch.tensor([True, False, True, False, False]),
  )


def test_follow_reference_averages_only_within_clip_in_heading_frame():
  zeros = torch.zeros(4)
  motion = SimpleNamespace(
    root_lin_vel_w=torch.tensor([[1.0, 0.0, 5.0]] * 2 + [[4.0, 0.0, 5.0]] * 2),
    root_ang_vel_w=torch.tensor([[2.0, 0.0, 0.5]] * 2 + [[2.0, 0.0, -0.8]] * 2),
    root_quat_w=quat_from_euler_xyz(
      zeros, torch.full_like(zeros, 0.6), torch.full_like(zeros, torch.pi / 2)
    ),
    motion_ends=torch.tensor([2, 4]),
  )
  task = make_velocity_task(
    (0.1, 1), (-1, 1), (-1, 1), 25, (0.5, 0.5, 0.7), mode="follow"
  )
  assert task.goal_specs[0].default == 0.1
  assert task.precompute_replay_goal is not None
  commands = task.precompute_replay_goal(motion)
  expected = torch.tensor([[0.0, -1.0, 0.5]] * 2 + [[0.0, -4.0, -0.8]] * 2)
  torch.testing.assert_close(commands, expected)
