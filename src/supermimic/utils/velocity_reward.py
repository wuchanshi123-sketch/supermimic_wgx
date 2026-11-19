"""Velocity-following reward and fall criteria from SuperMimic Table A.6."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

MIN_ROOT_HEIGHT = 0.35
MAX_TILT_ERROR = 0.7
VELOCITY_MAX_REWARD_RATE = 4.0


def heading_velocity(
  root_quat: torch.Tensor,
  lin_vel_w: torch.Tensor,
  ang_vel_w: torch.Tensor,
) -> torch.Tensor:
  """Pack heading-frame planar velocity and world-frame yaw rate."""
  planar = quat_apply_inverse(yaw_quat(root_quat), lin_vel_w)[..., :2]
  return torch.cat([planar, ang_vel_w[..., 2:3]], dim=-1)


def _root_tilt_error(root_quat: torch.Tensor) -> torch.Tensor:
  # 1 + gravity_z in the root frame, using world gravity direction [0, 0, -1].
  return 2.0 * root_quat[..., 1:3].square().sum(-1)


def velocity_fallen(root_pos: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
  return (root_pos[..., 2] < MIN_ROOT_HEIGHT) | (
    _root_tilt_error(root_quat) > MAX_TILT_ERROR
  )


def velocity_fall_probability(
  root_pos: torch.Tensor, root_quat: torch.Tensor, softness: float
) -> torch.Tensor:
  """Differentiable form of the absolute velocity-task fall criteria."""
  low = (MIN_ROOT_HEIGHT - root_pos[..., 2]) / MIN_ROOT_HEIGHT
  tilted = _root_tilt_error(root_quat) / MAX_TILT_ERROR - 1.0
  return torch.sigmoid(torch.maximum(low, tilted) / softness)


def velocity_follow_reward(
  command: torch.Tensor,
  root_quat: torch.Tensor,
  root_lin_vel_w: torch.Tensor,
  root_ang_vel_w: torch.Tensor,
  torso_rot_mat_w: torch.Tensor,
  torso_ang_vel_w: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_limits: torch.Tensor,
  action: torch.Tensor,
  last_action: torch.Tensor,
  dt: float,
) -> torch.Tensor:
  """Track the command applied on this step, with dt-scaled regularization."""
  error = heading_velocity(root_quat, root_lin_vel_w, root_ang_vel_w) - command
  linear = torch.exp(-error[..., :2].square().sum(-1) / 0.25)
  yaw = torch.exp(-error[..., 2].square() / 0.5)

  heading = yaw_quat(root_quat)
  torso_up = quat_apply_inverse(heading, torso_rot_mat_w[..., :, 2])
  torso_rates = quat_apply_inverse(heading, torso_ang_vel_w)
  lean = torso_up[..., 1].square()
  roll_pitch_rate = torso_rates[..., :2].square().sum(-1)

  action_rate = (action - last_action).square().sum(-1)
  lo, hi = joint_limits.unbind(-1)
  joint_limit = ((lo - joint_pos).clamp(min=0) + (joint_pos - hi).clamp(min=0)).sum(-1)
  return dt * (
    2.0 * linear
    + 2.0 * yaw
    - 0.3 * action_rate
    - 2.0 * joint_limit
    - 4.0 * lean
    - 0.05 * roll_pitch_rate
  )
