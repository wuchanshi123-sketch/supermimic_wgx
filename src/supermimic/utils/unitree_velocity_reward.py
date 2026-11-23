"""Non-foot rewards from Unitree's Unitree-G1-Flat environment.

Reference configuration and equations are from unitreerobotics/unitree_rl_mjlab
revision 1425b15f73bd4095f0df53709d7c389c3eb9e790.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from supermimic.utils.gencoord_helpers import decompose_rep_state

# Widths (walking, running) are keyed by G1 joint name fragments.
_POSTURE_WIDTHS = (
  ("hip_pitch", 0.5, 0.5),
  ("hip_roll", 0.15, 0.25),
  ("hip_yaw", 0.15, 0.25),
  ("knee", 0.5, 0.5),
  ("ankle_pitch", 0.15, 0.25),
  ("ankle_roll", 0.1, 0.1),
  ("waist_yaw", 0.15, 0.25),
  ("waist_roll", 0.1, 0.1),
  ("waist_pitch", 0.1, 0.1),
  ("shoulder_pitch", 0.15, 0.25),
  ("shoulder_roll", 0.1, 0.1),
  ("shoulder_yaw", 0.1, 0.1),
  ("elbow", 0.1, 0.1),
  ("wrist", 0.1, 0.1),
)


def unitree_g1_home_joint_pos(
  joint_names: Sequence[str], like: torch.Tensor
) -> torch.Tensor:
  """Return the official G1 HOME_KEYFRAME angles in the entity's joint order."""
  pair_angles = {
    "hip_pitch_joint": -0.1,
    "knee_joint": 0.3,
    "ankle_pitch_joint": -0.2,
    "shoulder_pitch_joint": 0.35,
    "elbow_joint": 0.87,
  }
  angles = {
    f"{side}_{name}": value
    for side in ("left", "right")
    for name, value in pair_angles.items()
  }
  angles.update(left_shoulder_roll_joint=0.18, right_shoulder_roll_joint=-0.18)
  return like.new_tensor(
    [angles.get(name.rsplit("/", 1)[-1], 0.0) for name in joint_names]
  )


@dataclass(frozen=True)
class UnitreeVelocityRewardCfg:
  linear_weight: float = 1.0
  yaw_weight: float = 1.0
  pose_weight: float = 1.0
  linear_std: float = 0.5
  yaw_std: float = math.sqrt(0.5)
  standing_pose_std: float = 0.05
  walking_threshold: float = 0.1
  running_threshold: float = 1.5
  stand_still_threshold: float = 0.1
  torso_orientation_cost: float = 1.0
  torso_roll_pitch_speed_cost: float = 0.05
  angular_momentum_cost: float = 0.025
  terminated_cost: float = 200.0
  joint_acceleration_cost: float = 2.5e-7
  joint_limit_cost: float = 10.0
  action_rate_cost: float = 0.05
  stand_still_cost: float = 1.0
  self_collision_cost: float = 1.0
  foot_slip_cost: float = 0.25
  soft_landing_cost: float = 1.0e-3
  foot_command_threshold: float = 0.1

  def __post_init__(self) -> None:
    widths = (self.linear_std, self.yaw_std, self.standing_pose_std)
    if any(not math.isfinite(width) or width <= 0.0 for width in widths):
      raise ValueError("Unitree velocity reward widths must be finite and positive")
    weights = (
      self.linear_weight,
      self.yaw_weight,
      self.pose_weight,
      self.torso_orientation_cost,
      self.torso_roll_pitch_speed_cost,
      self.angular_momentum_cost,
      self.terminated_cost,
      self.joint_acceleration_cost,
      self.joint_limit_cost,
      self.action_rate_cost,
      self.stand_still_cost,
      self.self_collision_cost,
      self.foot_slip_cost,
      self.soft_landing_cost,
    )
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
      raise ValueError(
        "Unitree velocity reward weights must be finite and non-negative"
      )
    if not 0.0 <= self.walking_threshold <= self.running_threshold < math.inf:
      raise ValueError("Invalid Unitree walking and running thresholds")
    if not 0.0 <= self.stand_still_threshold < math.inf:
      raise ValueError("Invalid Unitree stand-still threshold")
    if not 0.0 <= self.foot_command_threshold < math.inf:
      raise ValueError("Invalid Unitree foot reward command threshold")

  @property
  def max_reward(self) -> float:
    """Per-second upper bound used for critic targets."""
    return self.linear_weight + self.yaw_weight + self.pose_weight


class UnitreeVelocityPosture:
  """Nominal posture and per-joint widths in model joint order."""

  def __init__(
    self,
    joint_names: Sequence[str],
    default_joint_pos: torch.Tensor,
    *,
    joint_widths: Sequence[tuple[str, float, float]] = _POSTURE_WIDTHS,
  ) -> None:
    if default_joint_pos.shape[-1] != len(joint_names):
      raise ValueError("The Unitree posture target must match the robot's joint count")
    walking, running = [], []
    for name in joint_names:
      joint_name = name.rsplit("/", 1)[-1]
      for fragment, walk, run in joint_widths:
        if fragment in joint_name:
          walking.append(walk)
          running.append(run)
          break
      else:
        raise ValueError(f"No posture widths for joint {name!r}")
    self.default_joint_pos = default_joint_pos
    self.walking_std = default_joint_pos.new_tensor(walking)
    self.running_std = default_joint_pos.new_tensor(running)


def unitree_velocity_reward(
  rep_state: torch.Tensor,
  command: torch.Tensor,
  action: torch.Tensor,
  last_action: torch.Tensor,
  joint_limits: torch.Tensor,
  cfg: UnitreeVelocityRewardCfg,
  *,
  posture: UnitreeVelocityPosture,
  torso_tilt_error: torch.Tensor,
  torso_roll_pitch_speed_sq: torch.Tensor,
  joint_acc_l2: torch.Tensor,
  angular_momentum_l2: torch.Tensor,
  self_collision_count: torch.Tensor,
  foot_slip: torch.Tensor,
  soft_landing: torch.Tensor,
  terminated: torch.Tensor,
) -> torch.Tensor:
  """Score post-step physics or imagined values before control-period scaling.

  The torso tilt error is 1 - cos(tilt). The remaining extra signals are already
  reduced per environment. In imagination, the termination value can be an
  expected indicator in [0, 1].
  """
  _, _, lin_vel, ang_vel, joint_pos, _ = decompose_rep_state(rep_state)
  xy_error = (command[..., :2] - lin_vel[..., :2]).square().sum(-1)
  linear_error = xy_error + 2.0 * lin_vel[..., 2].square()
  yaw_error = (command[..., 2] - ang_vel[..., 2]).square()
  angular_error = yaw_error + 0.05 * ang_vel[..., :2].square().sum(-1)

  command_speed = (
    torch.linalg.vector_norm(command[..., :2], dim=-1) + command[..., 2].abs()
  )
  posture_std = torch.where(
    command_speed[..., None] < cfg.walking_threshold,
    cfg.standing_pose_std,
    torch.where(
      command_speed[..., None] < cfg.running_threshold,
      posture.walking_std,
      posture.running_std,
    ),
  )
  pose_sq_error = (joint_pos - posture.default_joint_pos).square()
  pose = torch.exp(-(pose_sq_error / posture_std.square()).mean(-1))
  stand_still = pose_sq_error.sum(-1) * (command_speed <= cfg.stand_still_threshold)
  foot_cost = torch.where(
    command_speed > cfg.foot_command_threshold,
    cfg.foot_slip_cost * foot_slip + cfg.soft_landing_cost * soft_landing,
    0.0,
  )

  tilt = torso_tilt_error.clamp(min=0.0, max=2.0)
  torso_orientation = tilt * (2.0 - tilt)  # XY norm squared of projected gravity.
  lower, upper = joint_limits.unbind(-1)
  joint_limit = ((lower - joint_pos).relu() + (joint_pos - upper).relu()).sum(-1)
  action_rate = (action - last_action).square().sum(-1)

  return (
    cfg.linear_weight * torch.exp(-linear_error / cfg.linear_std**2)
    + cfg.yaw_weight * torch.exp(-angular_error / cfg.yaw_std**2)
    + cfg.pose_weight * pose
    - cfg.torso_orientation_cost * torso_orientation
    - cfg.torso_roll_pitch_speed_cost * torso_roll_pitch_speed_sq
    - cfg.angular_momentum_cost * angular_momentum_l2
    - cfg.terminated_cost * terminated
    - cfg.joint_acceleration_cost * joint_acc_l2
    - cfg.joint_limit_cost * joint_limit
    - cfg.action_rate_cost * action_rate
    - cfg.stand_still_cost * stand_still
    - cfg.self_collision_cost * self_collision_count
    - foot_cost
  )
