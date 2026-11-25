"""Alternating steps and stable support for low-speed R00 turns."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch

# Direction, world XY anchor, previous contact (L/R), last valid step side,
# air time (L/R), peak clearance (L/R), supported time (L/R), and yaw progress (L/R).
_MEMORY_WIDTHS = (1, 2, 2, 1, 2, 2, 2, 2)
TURN_MEMORY_DIM = sum(_MEMORY_WIDTHS)


@dataclass(frozen=True)
class TurnRewardCfg:
  enabled: bool = False
  """Enable low-speed pure-turn shaping."""
  twist_weight: float = 0.30
  """Cost per squared rad/s of support-foot twist, weighted by load fraction."""
  clearance_weight: float = 0.25
  """Maximum reward rate for clearance with opposite-foot support."""
  landing_weight: float = 0.04
  """Bonus per valid alternating landing, before conversion to a reward rate."""
  min_clearance: float = 0.03
  target_clearance: float = 0.05
  min_swing_time: float = 0.30
  max_swing_time: float = 0.75
  min_step_yaw: float = 0.04
  drift_deadband: float = 0.06
  drift_time_scale: float = 0.60

  def __post_init__(self) -> None:
    weights = (self.twist_weight, self.clearance_weight, self.landing_weight)
    if any(not math.isfinite(w) or w < 0.0 for w in weights):
      raise ValueError("Turn reward weights must be finite and non-negative")
    if not 0.0 < self.min_clearance <= self.target_clearance < math.inf:
      raise ValueError("Turn clearance requires 0 < min_clearance <= target_clearance")
    if not 0.0 < self.min_swing_time < self.max_swing_time < math.inf:
      raise ValueError("Turn swing times must be positive, finite and ascending")
    if not 0.0 < self.min_step_yaw < math.inf:
      raise ValueError("Turn step yaw progress must be positive and finite")
    if not (
      0.0 <= self.drift_deadband < math.inf and 0.0 < self.drift_time_scale < math.inf
    ):
      raise ValueError("Turn drift deadband and time scale must be finite and valid")

  def max_bonus_per_second(self, dt: float) -> float:
    return self.clearance_weight + self.landing_weight / dt if self.enabled else 0.0


class TurnRewardTerms(NamedTuple):
  gate: torch.Tensor
  twist: torch.Tensor
  clearance: torch.Tensor
  alternating: torch.Tensor
  drift_error: torch.Tensor


def _smooth_ramp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
  u = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
  return u.square() * (3.0 - 2.0 * u)


def turn_command_gate(command: torch.Tensor, cfg: TurnRewardCfg) -> torch.Tensor:
  """Full weight for slow turns at 0.4–1.0 rad/s and negligible translation."""
  if not cfg.enabled:
    return torch.zeros_like(command[..., 2])
  linear_speed = torch.linalg.vector_norm(command[..., :2], dim=-1)
  yaw_speed = command[..., 2].abs()
  return (
    (1.0 - _smooth_ramp(linear_speed, 0.05, 0.10))
    * _smooth_ramp(yaw_speed, 0.30, 0.40)
    * (1.0 - _smooth_ramp(yaw_speed, 1.0, 1.2))
  )


def turn_critic_context(memory: torch.Tensor, root_xy: torch.Tensor) -> torch.Tensor:
  """Expose the current offset from the turn anchor in the critic input."""
  direction, anchor, tail = memory.split((1, 2, TURN_MEMORY_DIM - 3), dim=-1)
  offset = torch.where(direction != 0.0, root_xy - anchor, 0.0)
  return torch.cat((direction, offset, tail), dim=-1)


def evaluate_turn(
  memory: torch.Tensor,
  command: torch.Tensor,
  root_xy: torch.Tensor,
  yaw_rate: torch.Tensor,
  sole_height: torch.Tensor,
  contact: torch.Tensor,
  normal_load: torch.Tensor,
  foot_yaw_rate: torch.Tensor,
  dt: float,
  cfg: TurnRewardCfg,
) -> tuple[torch.Tensor, TurnRewardTerms]:
  """Advance gait memory and return unmasked turn terms in reward-rate units.

  Per-foot inputs use left/right order. Heights are metres above the plane,
  angular rates are world-normal components, and loads are fractions of robot
  weight. Contact is binary in physics and can be a smooth estimate in rollout.
  """
  gate = turn_command_gate(command, cfg)
  if not cfg.enabled:
    return memory, TurnRewardTerms(gate, gate, gate, gate, gate)
  contact = contact.to(memory.dtype)
  active = gate > 0.0
  direction = command[..., 2].sign()
  saved_direction, anchor, prev_ground, last, air, peak, supported, progress = (
    memory.split(_MEMORY_WIDTHS, dim=-1)
  )
  continuing = active & (saved_direction.squeeze(-1) == direction)
  continuing_foot = continuing.unsqueeze(-1)
  anchor = torch.where(continuing_foot, anchor, root_xy)
  last = torch.where(continuing_foot, last, 0.0)
  air = torch.where(continuing_foot, air, 0.0)
  peak = torch.where(continuing_foot, peak, 0.0)
  supported = torch.where(continuing_foot, supported, 0.0)
  progress = torch.where(continuing_foot, progress, 0.0)

  # Hysteresis prevents a barely touching foot from repeatedly landing.
  was_grounded = torch.where(continuing_foot, prev_ground > 0.5, contact >= 0.5)
  grounded = torch.where(was_grounded, contact >= 0.35, contact >= 0.65)
  landing_edge = grounded & ~was_grounded
  # Weight by the opposite foot's load. Thirty percent of body weight marks
  # full support for the swing, leaving room for natural weight transfers.
  support = (normal_load.flip(-1) / 0.30).clamp(0.0, 1.0) * contact.flip(-1)
  one = torch.ones_like(last)
  side = torch.cat((-one, one), dim=-1)
  expected_side = last != side
  yaw_advance = dt * (direction * yaw_rate).unsqueeze(-1)
  valid = (
    landing_edge
    & continuing_foot
    & ~landing_edge.all(dim=-1, keepdim=True)
    & grounded.flip(-1)
    & expected_side
    & (air >= cfg.min_swing_time - 1e-6)
    & (air <= cfg.max_swing_time + 1e-6)
    & (peak >= cfg.min_clearance)
    & (supported >= air * 0.75)
    & (progress + yaw_advance >= cfg.min_step_yaw)
  )
  landed = (valid * side).sum(dim=-1, keepdim=True)
  last = torch.where(landed != 0.0, landed, last)

  elapsed = air + dt
  # Reward the start of a lift as well as reaching the target, and expire
  # the reward for a held foot. A completed step gets a separate bonus.
  lift = (sole_height / cfg.target_clearance).clamp(0.0, 1.0)
  expired = _smooth_ramp(elapsed, cfg.max_swing_time - dt, cfg.max_swing_time + dt)
  clearance = (
    (lift * (1.0 - contact) * support * (1.0 - expired) * expected_side)
    .sum(dim=-1)
    .clamp(max=1.0)
  )
  alternating = (valid * contact).sum(dim=-1) * (cfg.landing_weight / dt)
  twist = -(normal_load.clamp(0.0, 1.0) * foot_yaw_rate.square()).sum(dim=-1)

  # Root translation is measured against a world-frame turn center, so
  # changing the yaw cannot cancel drift in changing body coordinates.
  displacement = torch.linalg.vector_norm(root_xy - anchor, dim=-1)
  drift_error = (
    (displacement - cfg.drift_deadband).relu() / cfg.drift_time_scale
  ).square()

  next_memory = torch.cat(
    (
      direction.unsqueeze(-1),
      anchor,
      grounded.to(memory.dtype),
      last,
      torch.where(grounded, 0.0, elapsed),
      torch.where(grounded, 0.0, torch.maximum(peak, sole_height)),
      torch.where(grounded, 0.0, supported + dt * support),
      torch.where(grounded, 0.0, progress + yaw_advance),
    ),
    dim=-1,
  )
  return torch.where(active.unsqueeze(-1), next_memory, 0.0), TurnRewardTerms(
    gate,
    cfg.twist_weight * twist,
    cfg.clearance_weight * clearance,
    alternating,
    drift_error,
  )
