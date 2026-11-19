"""Fall criteria for the G1 velocity task."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from supermimic.utils.gencoord_helpers import decompose_rep_state


@dataclass(frozen=True)
class VelocityTerminationCfg:
  min_height: float = 0.45
  max_tilt_rad: float = math.radians(60.0)

  def __post_init__(self) -> None:
    if not math.isfinite(self.min_height) or self.min_height <= 0.0:
      raise ValueError("Velocity termination height must be finite and positive")
    if not 0.0 < self.max_tilt_rad < math.pi:
      raise ValueError("Velocity termination tilt must be between 0 and pi radians")


def velocity_fall_error(
  rep_state: torch.Tensor, cfg: VelocityTerminationCfg
) -> torch.Tensor:
  """Normalized fall error, with a threshold of one for both criteria."""
  height, gravity, *_ = decompose_rep_state(rep_state)
  height_error = 2.0 - height.squeeze(-1) / cfg.min_height
  tilt_error = (1.0 + gravity[..., 2]) / (1.0 - math.cos(cfg.max_tilt_rad))
  return torch.maximum(height_error, tilt_error)
