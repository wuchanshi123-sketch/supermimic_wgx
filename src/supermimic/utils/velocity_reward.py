"""Pure velocity rewards and fall criteria for G1 generalized coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from supermimic.utils.gencoord_helpers import decompose_rep_state


@dataclass(frozen=True)
class VelocityRewardCfg:
  linear_weight: float = 2.0
  yaw_weight: float = 2.0
  upright_weight: float = 0.5
  linear_std: float = 0.5
  yaw_std: float = 0.75
  upright_std: float = 0.4
  vertical_speed_cost: float = 0.4
  roll_pitch_speed_cost: float = 0.05
  action_rate_cost: float = 0.1
  joint_limit_cost: float = 1.0
  standing_height_cost: float = 2.0
  standing_height: float = 0.76

  def __post_init__(self) -> None:
    widths = (self.linear_std, self.yaw_std, self.upright_std)
    if any(not math.isfinite(x) or x <= 0.0 for x in widths):
      raise ValueError(
        "Velocity reward standard deviations must be finite and positive"
      )
    weights = (
      self.linear_weight,
      self.yaw_weight,
      self.upright_weight,
      self.vertical_speed_cost,
      self.roll_pitch_speed_cost,
      self.action_rate_cost,
      self.joint_limit_cost,
      self.standing_height_cost,
    )
    if any(not math.isfinite(x) or x < 0.0 for x in weights):
      raise ValueError("Velocity reward weights must be finite and non-negative")

  @property
  def max_reward(self) -> float:
    """Per-second upper bound used for critic targets."""
    return self.linear_weight + self.yaw_weight + self.upright_weight


@dataclass(frozen=True)
class VelocityTerminationCfg:
  min_height: float = 0.45
  max_tilt_rad: float = math.radians(60.0)

  def __post_init__(self) -> None:
    if not math.isfinite(self.min_height) or self.min_height <= 0.0:
      raise ValueError("Velocity termination height must be finite and positive")
    if not 0.0 < self.max_tilt_rad < math.pi:
      raise ValueError("Velocity termination tilt must be between 0 and pi radians")


def velocity_reward(
  rep_state: torch.Tensor,
  command: torch.Tensor,
  action: torch.Tensor,
  last_action: torch.Tensor,
  joint_limits: torch.Tensor,
  cfg: VelocityRewardCfg,
) -> torch.Tensor:
  """Reward rate from the post-step state, the command and raw policy actions."""
  height, gravity, lin_vel, ang_vel, joint_pos, _ = decompose_rep_state(rep_state)
  linear_error = (lin_vel[..., :2] - command[..., :2]).square().sum(-1)
  yaw_error = (ang_vel[..., 2] - command[..., 2]).square()
  upright_error = (1.0 + gravity[..., 2]).clamp_min(0.0)
  tracking = cfg.linear_weight * torch.exp(-linear_error / cfg.linear_std**2)
  tracking = tracking + cfg.yaw_weight * torch.exp(-yaw_error / cfg.yaw_std**2)
  upright = cfg.upright_weight * torch.exp(-upright_error / cfg.upright_std**2)

  vertical_motion = lin_vel[..., 2].square()
  roll_pitch = ang_vel[..., :2].square().sum(-1)
  action_rate = (action - last_action).square().sum(-1)
  lower, upper = joint_limits.unbind(-1)
  joint_limit = ((lower - joint_pos).relu() + (joint_pos - upper).relu()).sum(-1)
  standing = (command == 0.0).all(dim=-1)
  standing_height = (height.squeeze(-1) - cfg.standing_height).square() * standing

  return (
    tracking
    + upright
    - cfg.vertical_speed_cost * vertical_motion
    - cfg.roll_pitch_speed_cost * roll_pitch
    - cfg.action_rate_cost * action_rate
    - cfg.joint_limit_cost * joint_limit
    - cfg.standing_height_cost * standing_height
  )


def velocity_fall_error(
  rep_state: torch.Tensor, cfg: VelocityTerminationCfg
) -> torch.Tensor:
  """Normalized fall error, with a threshold of one for both criteria."""
  height, gravity, *_ = decompose_rep_state(rep_state)
  height_error = 2.0 - height.squeeze(-1) / cfg.min_height
  tilt_error = (1.0 + gravity[..., 2]) / (1.0 - math.cos(cfg.max_tilt_rad))
  return torch.maximum(height_error, tilt_error)
