"""Load clip-safe G1 expert transitions for AMP velocity training."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import torch


def _resolve_motion_files(source: str | Path) -> list[Path]:
  path = Path(source).expanduser()
  matches: Iterable[Path]
  if path.is_file():
    matches = [path]
  elif path.is_dir():
    matches = path.iterdir()
  else:
    matches = map(Path, glob(str(path)))
  files = sorted(p for p in matches if p.is_file() and p.suffix.lower() == ".npz")
  if not files:
    raise ValueError(f"No AMP reference .npz files found for {source!s}")
  return files


def _load_clip(path: Path, joints: tuple[str, ...], dt: float) -> np.ndarray:
  with np.load(path, allow_pickle=False) as motion:
    fps = float(motion["fps"].item())
    if not math.isfinite(fps) or not math.isclose(fps * dt, 1.0, rel_tol=1e-5):
      raise ValueError(
        f"expected {1.0 / dt:g} Hz to match the control step, got {fps:g} Hz"
      )

    names = motion["joint_names"]
    if names.ndim != 1 or names.dtype.kind not in "US":
      raise ValueError("joint_names must be a one-dimensional string array")
    columns = {name: idx for idx, name in enumerate(names.astype(str).tolist())}
    if len(columns) != len(names) or columns.keys() != set(joints):
      raise ValueError("joint_names must match the robot's joints with no duplicates")

    num_frames = len(motion["joint_pos"])
    if num_frames < 2:
      raise ValueError("each AMP clip must have at least two frames")

    # Keep the same layout as the simulator's GenCoord full_state observation.
    layout = (
      ("root_pos_w", 3),
      ("root_quat_w", 4),
      ("joint_pos", len(joints)),
      ("root_lin_vel_w", 3),
      ("root_ang_vel_w", 3),
      ("joint_vel", len(joints)),
    )
    arrays = {}
    for name, width in layout:
      values = motion[name]
      if values.shape != (num_frames, width) or values.dtype.kind not in "fiu":
        raise ValueError(
          f"{name} must be a real array with shape ({num_frames}, {width})"
        )
      arrays[name] = values.astype(np.float32, copy=False)
      if not np.isfinite(arrays[name]).all():
        raise ValueError(f"{name} contains non-finite values")

    quat = arrays["root_quat_w"]
    norm = np.linalg.norm(quat.astype(np.float64), axis=1, keepdims=True)
    if np.any(norm < 1e-8):
      raise ValueError("root_quat_w contains a zero quaternion")
    arrays["root_quat_w"] = (quat / norm).astype(np.float32)

    order = [columns[joint] for joint in joints]
    for name in ("joint_pos", "joint_vel"):
      arrays[name] = arrays[name][:, order]
    return np.concatenate([arrays[name] for name, _ in layout], axis=-1)


@dataclass(frozen=True)
class AmpReference:
  """Robot-order full states and in-clip starts for two-frame transitions."""

  states: torch.Tensor
  transition_starts: torch.Tensor
  clip_starts: torch.Tensor
  clip_lengths: torch.Tensor

  @property
  def num_clips(self) -> int:
    return self.clip_starts.numel()

  @property
  def num_frames(self) -> int:
    return self.states.shape[0]

  @property
  def num_transitions(self) -> int:
    return self.transition_starts.numel()

  @classmethod
  def load(
    cls,
    motion_file: str | Path,
    joint_names: Sequence[str],
    dt: float,
    device: str | torch.device,
  ) -> AmpReference:
    if not math.isfinite(dt) or dt <= 0.0:
      raise ValueError("The AMP control step must be finite and positive")
    joints = tuple(name.rsplit("/", 1)[-1] for name in joint_names)
    if not joints or len(set(joints)) != len(joints):
      raise ValueError("The robot's AMP joint names must be nonempty and unique")

    files = _resolve_motion_files(motion_file)
    clips, starts = [], []
    boundaries = [0]
    for path in files:
      try:
        clip = _load_clip(path, joints, dt)
      except (BadZipFile, KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid AMP reference {path}: {exc}") from exc
      clips.append(clip)
      start = boundaries[-1]
      end = start + len(clip)
      starts.append(np.arange(start, end - 1, dtype=np.int64))
      boundaries.append(end)

    states = torch.from_numpy(np.concatenate(clips)).to(device=device)
    transition_starts = torch.from_numpy(np.concatenate(starts)).to(device=device)
    boundaries = np.asarray(boundaries, dtype=np.int64)
    clip_starts = torch.from_numpy(boundaries[:-1]).to(device=device)
    clip_lengths = torch.from_numpy(np.diff(boundaries)).to(device=device)
    return cls(states, transition_starts, clip_starts, clip_lengths)

  def sample_indices(self, batch_size: int) -> torch.Tensor:
    """Sample clips uniformly, then sample adjacent-frame starts within each clip."""
    device = self.states.device
    clips = torch.randint(self.num_clips, (batch_size,), device=device)
    num_pairs = self.clip_lengths[clips] - 1
    offsets = (torch.rand(batch_size, device=device) * num_pairs).long()
    return self.clip_starts[clips] + offsets.minimum(num_pairs - 1)

  def sample(self, batch_size: int) -> torch.Tensor:
    """Return ``[batch_size, 2, full_state_dim]`` consecutive expert frames."""
    starts = self.sample_indices(batch_size)
    return torch.stack((self.states[starts], self.states[starts + 1]), dim=1)
