"""Retime the 50 Hz WalkandRun G1 clips for velocity-following training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

FPS = 50
WINDOW = 25
AXES = ("vx", "vy", "wz")
LIMITS = np.array([1.5, 0.5, 3.14])
# Target speeds for a family of clips that already moves along the specified axis.
TARGETS = (
  (0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95, 1.10, 1.25, 1.40, 1.50),
  (0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
  (0.20, 0.40, 0.65, 0.90, 1.15, 1.40, 1.65, 1.90, 2.15, 2.40, 2.65, 2.90, 3.14),
)
TOLERANCES = (0.06, 0.035, 0.15)
SOURCE_PREFIXES = (
  ("walk_backward", "walk_forward"),
  ("walk_sideway_right", "walk_sideway_left"),
  ("idle_turn_", "idle_turn_"),
)


def reference_commands(clip: dict[str, np.ndarray], window: int = WINDOW) -> np.ndarray:
  """Match the velocity-following reference labels within one clip."""
  twist = np.column_stack((clip["root_lin_vel_w"], clip["root_ang_vel_w"][:, 2]))
  half = window // 2
  padded = np.pad(twist.astype(np.float64), ((half, half), (0, 0)), mode="edge")
  summed = np.vstack((np.zeros((1, 4)), padded.cumsum(axis=0)))
  smooth = (summed[window:] - summed[:-window]) / window
  w, x, y, z = clip["root_quat_w"].astype(np.float64).T
  yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  c, s = np.cos(yaw), np.sin(yaw)
  vx, vy = smooth[:, :2].T
  return np.column_stack((c * vx + s * vy, -s * vx + c * vy, smooth[:, 3]))


def resample_poses(
  clip: dict[str, np.ndarray], times: np.ndarray
) -> dict[str, np.ndarray]:
  """Keep the pose trajectory and recompute twists on the new 50 Hz grid."""
  original = np.arange(len(clip["joint_pos"]), dtype=np.float64)
  frames = len(times)
  result = {name: clip[name].copy() for name in ("fps", "joint_names")}
  for pose, velocity in (("root_pos_w", "root_lin_vel_w"), ("joint_pos", "joint_vel")):
    positions = np.column_stack(
      [np.interp(times, original, axis) for axis in clip[pose].T]
    ).astype(np.float32)
    result[pose] = positions
    result[velocity] = np.gradient(
      positions.astype(np.float64), 1 / FPS, axis=0
    ).astype(np.float32)

  old_rot = Rotation.from_quat(clip["root_quat_w"], scalar_first=True)
  new_rot = Slerp(original, old_rot)(times)
  quat = new_rot.as_quat(scalar_first=True)
  flip = np.sum(quat[:-1] * quat[1:], axis=1) < 0
  quat[1:] *= np.cumprod(np.where(flip, -1, 1))[:, None]
  result["root_quat_w"] = quat.astype(np.float32)
  # Relative rotation R_after * inv(R_before) expresses the angular velocity in world axes.
  i = np.arange(frames)
  before, after = np.maximum(i - 1, 0), np.minimum(i + 1, frames - 1)
  result["root_ang_vel_w"] = (
    (new_rot[after] * new_rot[before].inv()).as_rotvec()
    * (FPS / (after - before))[:, None]
  ).astype(np.float32)
  return result


def retime(
  clip: dict[str, np.ndarray], factor: float
) -> tuple[dict[str, np.ndarray], float]:
  source_intervals = len(clip["joint_pos"]) - 1
  intervals = max(1, round(source_intervals / factor))
  times = np.linspace(0, source_intervals, intervals + 1)
  return resample_poses(clip, times), source_intervals / intervals


def longest_run(mask: np.ndarray) -> tuple[int, int]:
  edges = np.diff(np.r_[False, mask, False].astype(np.int8))
  starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
  if not len(starts):
    return 0, 0
  best = np.argmax(stops - starts)
  return int(starts[best]), int(stops[best])


def active_part(
  clip: dict[str, np.ndarray], axis: int, sign: int
) -> tuple[dict[str, np.ndarray], int, int]:
  """Crop idle tails from an augmentation source, leaving the gait cycles intact."""
  speed = sign * reference_commands(clip)[:, axis]
  start, stop = longest_run(speed > 0.35 * np.percentile(speed, 90))
  if stop - start < FPS // 2:
    return clip, 0, len(speed)
  start, stop = max(0, start - 4), min(len(speed), stop + 4)
  part = {
    key: value if key in ("fps", "joint_names") else value[start:stop]
    for key, value in clip.items()
  }
  return part, start, stop


def fit_to_limits(
  clip: dict[str, np.ndarray], factor: float
) -> tuple[dict[str, np.ndarray], float, np.ndarray]:
  """Slow the retime if another axis would leave the command envelope."""
  for _ in range(8):
    candidate, actual = retime(clip, factor)
    commands = reference_commands(candidate)
    peak = (np.abs(commands) / LIMITS).max()
    if peak <= 1.0:
      return candidate, actual, commands
    factor *= 0.997 / peak
  raise ValueError("Unable to retime a clip inside the commanded speed envelope")


def steady_endpoint(originals, axis: int, sign: int, prefix: str):
  """Find a continuous segment that stays near the envelope edge."""
  limit, tolerance = LIMITS[axis], TOLERANCES[axis]
  best = None
  for name, clip in originals.items():
    if not name.startswith(prefix):
      continue
    smoothed = sign * reference_commands(clip)[:, axis]
    peak = np.percentile(smoothed, 90)
    if peak < 0.1:
      continue
    raw = sign * reference_commands(clip, window=1)[:, axis]
    for threshold in (0.45, 0.55, 0.65, 0.75):
      first, stop = longest_run(smoothed > threshold * peak)
      if stop - first < WINDOW - 1:
        continue
      part = {
        key: value if key in ("fps", "joint_names") else value[first:stop]
        for key, value in clip.items()
      }
      frames = np.arange(stop - first)
      for sigma in (0.7, 1.5, 2.5, 4.0):
        # Use distance along the source axis as the clock. This flattens the
        # primary speed without changing where the source places its feet.
        speed = np.maximum(gaussian_filter1d(raw[first:stop], sigma), peak * 0.25)
        progress = np.r_[0, ((speed[:-1] + speed[1:]) / (2 * FPS)).cumsum()]
        for share in (0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99):
          intervals = max(1, round(progress[-1] / (share * limit) * FPS))
          times = np.interp(
            np.linspace(0, progress[-1], intervals + 1), progress, frames
          )
          variant = resample_poses(part, times)
          commands = reference_commands(variant)
          if (np.abs(commands) > LIMITS).any():
            continue
          a, b = longest_run(sign * commands[:, axis] >= limit - tolerance)
          cross_axis = np.delete(commands[a:b], axis, axis=1)
          secondary = (
            float(np.abs(cross_axis).mean()) if len(cross_axis) else float("inf")
          )
          score = b - a, -secondary
          if best is None or score > best[0]:
            best = score, name, variant, commands, first, stop, times
  return best


def coverage(labels: np.ndarray) -> dict:
  result = {}
  indices = []
  for col, (name, limit, width) in enumerate(
    zip(AXES, LIMITS, (0.1, 0.05, 0.2), strict=True)
  ):
    edges = np.linspace(-limit, limit, round(2 * limit / width) + 1)
    counts, _ = np.histogram(labels[:, col], bins=edges)
    result[name] = {
      "range": [float(labels[:, col].min()), float(labels[:, col].max())],
      "bin_edges": edges.tolist(),
      "bin_counts": counts.tolist(),
    }
    joint_edges = np.linspace(-limit, limit, 6)
    indices.append(
      np.clip(np.searchsorted(joint_edges, labels[:, col], side="right") - 1, 0, 4)
    )
  occupied = np.unique(np.column_stack(indices), axis=0)
  result["joint_5x5x5"] = {
    "occupied": len(occupied),
    "total": 125,
    "cells": occupied.tolist(),
  }
  return result


def generate(source: Path, output: Path) -> dict:
  if output.exists():
    raise FileExistsError(f"Output directory already exists: {output}")
  originals = {}
  joint_names = None
  for path in sorted(source.glob("*.npz")):
    with np.load(path, allow_pickle=False) as data:
      clip = {key: data[key] for key in data.files}
    if float(clip["fps"].item()) != FPS:
      raise ValueError(f"Expected 50 Hz input: {path}")
    if joint_names is None:
      joint_names = clip["joint_names"]
    if not np.array_equal(joint_names, clip["joint_names"]):
      raise ValueError(f"Mismatched joint order: {path}")
    originals[path.stem] = clip
  if not originals:
    raise ValueError(f"No .npz motions found in {source}")

  output.parent.mkdir(parents=True, exist_ok=True)
  records = []
  all_labels = []
  with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as tmp:
    stage = Path(tmp)

    def write(name, original, clip, commands, factor, **metadata):
      path = stage / f"{name}.npz"
      np.savez_compressed(path, **clip)
      all_labels.append(commands)
      records.append(
        {
          "name": path.name,
          "source": f"{original}.npz",
          "frames": len(commands),
          "source_speed_multiplier": float(factor),
          "joint_speed_max_rad_s": float(np.abs(clip["joint_vel"]).max()),
          **metadata,
        }
      )

    for name, clip in originals.items():
      commands = reference_commands(clip)
      factor = 1.0
      if (np.abs(commands) > LIMITS).any():
        clip, factor, commands = fit_to_limits(clip, factor)
      write(f"base__{name}", name, clip, commands, factor)

    for axis, (axis_name, targets, tolerance, prefixes) in enumerate(
      zip(AXES, TARGETS, TOLERANCES, SOURCE_PREFIXES, strict=True)
    ):
      for sign, prefix in zip((-1, 1), prefixes, strict=True):
        sources = []
        for name, clip in originals.items():
          if not name.startswith(prefix):
            continue
          speed = sign * reference_commands(clip)[:, axis]
          if np.percentile(speed, 90) < 0.1:
            continue
          part, first, last = active_part(clip, axis, sign)
          typical = np.median(sign * reference_commands(part)[:, axis])
          sources.append((name, part, float(typical), first, last))
        if not sources:
          raise ValueError(f"Missing {axis_name} reference for sign {sign}")

        for target in targets:
          best = None
          for name, part, typical, first, last in sources:
            for adjust in (0.78, 0.86, 0.93, 1.0, 1.08, 1.17, 1.28):
              requested = min(2.25, max(0.10, target / typical * adjust))
              clip, factor, commands = fit_to_limits(part, requested)
              speeds = sign * commands[:, axis]
              near = np.abs(speeds - target) <= tolerance
              start, stop = longest_run(near)
              secondary = np.delete(commands, axis, axis=1)
              cross_axis = np.abs(secondary).mean()
              score = (
                min(stop - start, 2 * FPS),
                -float(np.abs(speeds - target).mean()),
                -float(cross_axis),
              )
              if best is None or score > best[0]:
                best = score, name, clip, factor, commands, first, last, stop - start
          assert best is not None
          _, original, clip, factor, commands, first, last, sustained = best
          direction = "pos" if sign > 0 else "neg"
          write(
            f"{axis_name}_{direction}_{target:.2f}__{original}",
            original,
            clip,
            commands,
            factor,
            target_axis=axis_name,
            target_value=sign * target,
            tolerance=tolerance,
            longest_near_target_s=sustained / FPS,
            source_frames=[first, last],
          )

        steady = steady_endpoint(originals, axis, sign, prefix)
        if steady is not None:
          score, original, clip, commands, first, stop, times = steady
          write(
            f"steady_{axis_name}_{direction}__{original}",
            original,
            clip,
            commands,
            (stop - first - 1) / (len(times) - 1),
            target_axis=axis_name,
            target_value=sign * LIMITS[axis],
            tolerance=tolerance,
            longest_near_target_s=score[0] / FPS,
            source_frames=[first, stop],
            retiming="primary-axis progress clock",
            local_speed_multiplier_range=[
              float(np.diff(times).min()),
              float(np.diff(times).max()),
            ],
          )

    for name, clip in originals.items():
      if not name.startswith(("arc_walk_", "walk_arc_")):
        continue
      for rate in (0.4, 0.6, 0.8, 1.2):
        variant, actual, commands = fit_to_limits(clip, rate)
        write(f"arc_{rate:.2f}__{name}", name, variant, commands, actual)

    labels = np.concatenate(all_labels)
    report = {
      "format": "native_supermimic_root_joint_npz",
      "fps": FPS,
      "source": str(source.resolve()),
      "total_frames": len(labels),
      "seconds": len(labels) / FPS,
      "labels": "25-frame centered mean world twist per clip, planar velocity in heading frame",
      "target_limits_vx_vy_wz": LIMITS.tolist(),
      "coverage": coverage(labels),
      "scope": "Time retiming preserves the source trajectory. Coverage is kinematic, not a validation of stable gait for all 3D command combinations.",
      "clips": records,
    }
    (stage / "manifest.json").write_text(
      json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    stage.rename(output)
  return report


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-dir", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  args = parser.parse_args()
  report = generate(args.input_dir, args.output_dir)
  print(
    f"Saved {len(report['clips'])} clips, {report['total_frames']} frames to {args.output_dir}"
  )
  for axis in AXES:
    stats = report["coverage"][axis]
    bins = stats["bin_counts"]
    lo, hi = stats["range"]
    print(f"{axis}: [{lo:.3f}, {hi:.3f}], {sum(n > 0 for n in bins)}/{len(bins)} bins")
  print("Joint command bins:", report["coverage"]["joint_5x5x5"]["occupied"], "/ 125")


if __name__ == "__main__":
  main()
