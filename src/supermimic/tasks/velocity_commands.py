"""Reference and independent commands for velocity following."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from supermimic.tasks.task_spec import VelocityTaskCfg


def reference_bc_mask(
  reference_source: torch.Tensor,
  motion_times: torch.Tensor,
  motion_ends: torch.Tensor,
) -> torch.Tensor:
  """Use reference imitation only while the clip can advance."""
  clip = torch.searchsorted(motion_ends, motion_times, right=True)
  last_frame = motion_ends[clip] - 1
  return reference_source & (motion_times < last_frame)


class VelocityCommands:
  """Renew velocity commands using fixed-shape, device-side sampling."""

  def __init__(
    self,
    cfg: VelocityTaskCfg,
    reference: torch.Tensor,
    batch_size: int,
    dt: float,
    device: str,
  ) -> None:
    if not math.isfinite(dt) or dt <= 0.0:
      raise ValueError("control timestep must be finite and positive")
    if batch_size < 1:
      raise ValueError("velocity command batch must have at least one row")
    self.reference = reference.to(device)
    self._batch_size = batch_size
    self._device = device
    self._hold_steps = max(1, round(cfg.hold_seconds / dt))
    self._reference_probability = cfg.reference_probability
    self._standing_probability = cfg.standing_probability

    ranges = (cfg.lin_vel_x_range, cfg.lin_vel_y_range, cfg.ang_vel_z_range)
    self._lower = self.reference.new_tensor([lo for lo, _ in ranges])
    upper = self.reference.new_tensor([hi for _, hi in ranges])
    # A fixed axis has just one distinct bin even if configured with more.
    bins = [
      n if lo < hi else 1 for n, (lo, hi) in zip(cfg.command_bins, ranges, strict=True)
    ]
    self._bin_width = (upper - self._lower) / self.reference.new_tensor(bins)
    self._cells = torch.cartesian_prod(*(torch.arange(n, device=device) for n in bins))

  def _sample(
    self, active: torch.Tensor | None
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_and_stop = torch.rand((self._batch_size, 2), device=self._device)
    reference_source = source_and_stop[:, 0] < self._reference_probability
    standing = source_and_stop[:, 1] < self._standing_probability
    moving = ~reference_source & ~standing
    draw = moving if active is None else moving & active

    # Assign a shuffled cycle of bins to the rows actually drawing a moving
    # command. Ranking the mask keeps shapes fixed during CUDA graph capture.
    cell_count = self._cells.shape[0]
    order = torch.randperm(cell_count, device=self._device)
    rank = draw.cumsum(0) - 1
    cell = self._cells[order[rank.remainder(cell_count)]]
    jitter = torch.rand_like(cell, dtype=self.reference.dtype)
    fixed = self._lower + (cell + jitter) * self._bin_width
    fixed = torch.where(moving[:, None], fixed, 0.0)
    remaining = torch.full_like(reference_source, self._hold_steps, dtype=torch.long)
    return fixed, reference_source, remaining

  def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return initial fixed commands, reference-source flags and step counts."""
    return self._sample(None)

  def advance(
    self,
    fixed: torch.Tensor,
    reference_source: torch.Tensor,
    remaining: torch.Tensor,
    reset: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Consume one step and renew expired or reset environments."""
    renew = remaining <= 1
    if reset is not None:
      renew = renew | reset.bool()
    new_fixed, new_source, new_remaining = self._sample(renew)
    return (
      torch.where(renew[:, None], new_fixed, fixed),
      torch.where(renew, new_source, reference_source),
      torch.where(renew, new_remaining, remaining - 1),
    )

  def goal(
    self,
    fixed: torch.Tensor,
    reference_source: torch.Tensor,
    motion_times: torch.Tensor,
  ) -> torch.Tensor:
    """Read the current reference frame or hold the independent command."""
    return torch.where(reference_source[:, None], self.reference[motion_times], fixed)
