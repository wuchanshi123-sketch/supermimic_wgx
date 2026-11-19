"""Per-task specs for the multi-encoder actor.

A ``TaskSpec`` describes the goal input for one task encoder, with functions
for deriving reference goals from motion and processing user commands.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
from tensordict import TensorDict

from supermimic.representations import actor_goal_dim

if TYPE_CHECKING:
  from mjlab.tasks.tracking.mdp.commands import MotionLoader

  from supermimic.representations import StateRepresentation


@dataclass(frozen=True)
class ContinuousGoalSpec:
  """Slider control for one raw-goal dim (viewer UI)."""

  name: str
  min: float
  max: float
  step: float
  default: float


@dataclass(frozen=True)
class DiscreteGoalSpec:
  """Dropdown control for one raw-goal dim with a fixed value set."""

  name: str
  choices: tuple[float, ...]
  default: float


type GoalSpec = ContinuousGoalSpec | DiscreteGoalSpec


ExtractFromMotion = Callable[["MotionLoader", torch.Tensor, torch.Tensor], torch.Tensor]
BuildInferenceGoalObs = Callable[[torch.Tensor], torch.Tensor]
PrecomputeReplayGoal = Callable[["MotionLoader"], torch.Tensor]


@dataclass(frozen=True)
class TaskSpec:
  """Contract for one task slot of the multi-encoder actor."""

  name: str
  goal_dim: int
  extract_training_goal: ExtractFromMotion
  goal_specs: tuple[GoalSpec, ...] = ()
  build_inference_goal_obs: BuildInferenceGoalObs | None = None
  raw_goal_dim: int = 0
  precompute_replay_goal: PrecomputeReplayGoal | None = None
  """Optional noise-free per-frame command table for ONNX-export sanity
  checks and viewer replay mode. Returns shape ``(num_frames, raw_goal_dim)``."""


def build_actor_obs_td(
  goals_by_name: dict[str, torch.Tensor],
  history: torch.Tensor,
  critic: torch.Tensor | None = None,
) -> TensorDict:
  """``history`` (B, H, obs_dim) (+ optional ``critic``) plus one ``goal_<name>``
  key per task. The newest frame ``history[:, -1]`` is the current obs."""
  obs: dict[str, torch.Tensor] = {"history": history}
  if critic is not None:
    obs["critic"] = critic
  obs.update({f"goal_{name}": goal for name, goal in goals_by_name.items()})
  return TensorDict(obs, batch_size=[history.shape[0]], device=history.device)


def make_motion_task(representation: "StateRepresentation", num_goals: int) -> TaskSpec:
  """Motion-tracking task: goal is the full motion target (actor side)."""

  def extract(motion, motion_times, aux_state):
    _, actor_goal = representation.compute_wm_goal(motion, motion_times, aux_state)
    return actor_goal

  return TaskSpec(
    name="motion",
    goal_dim=actor_goal_dim(num_goals),
    extract_training_goal=extract,
  )


def make_velocity_task(
  lin_vel_x_range: tuple[float, float],
  lin_vel_y_range: tuple[float, float],
  ang_vel_z_range: tuple[float, float],
  window_frames: int,
  command_noise: tuple[float, float, float],
  device: str = "cpu",
  *,
  mode: Literal["imitation", "follow"] = "imitation",
) -> TaskSpec:
  """Build the three-channel velocity goal and its reference replay table.

  Imitation mode projects the reference into the robot body frame and adds
  uniform noise to the training command. Follow mode uses planar velocity in
  the reference heading frame and world yaw rate, averaged within the clip.
  """
  lo = torch.tensor(
    [lin_vel_x_range[0], lin_vel_y_range[0], ang_vel_z_range[0]], device=device
  )
  hi = torch.tensor(
    [lin_vel_x_range[1], lin_vel_y_range[1], ang_vel_z_range[1]], device=device
  )
  noise_half = torch.tensor(command_noise, device=device)

  def _clamp(cmd):
    return torch.clamp(cmd, lo.to(cmd.device), hi.to(cmd.device))

  def _centered_mean(track, motion_times, motion_ends=None):
    if window_frames <= 1:
      return track[motion_times]
    half = window_frames // 2
    offsets = (
      torch.arange(window_frames, device=motion_times.device, dtype=motion_times.dtype)
      - half
    )
    idx = motion_times.unsqueeze(-1) + offsets
    if motion_ends is None:
      idx = idx.clamp(min=0, max=track.shape[0] - 1)
    else:
      clip = torch.searchsorted(motion_ends, motion_times, right=True)
      first = torch.cat([motion_ends.new_zeros(1), motion_ends[:-1]])[clip]
      last = motion_ends[clip] - 1
      idx = torch.clamp(idx, min=first.unsqueeze(-1), max=last.unsqueeze(-1))
    return track[idx].mean(dim=1)

  def _body_frame_cmd(motion, times, quat):
    v_b = quat_apply_inverse(quat, _centered_mean(motion.root_lin_vel_w, times))
    w_b = quat_apply_inverse(quat, _centered_mean(motion.root_ang_vel_w, times))
    return torch.stack([v_b[:, 0], v_b[:, 1], w_b[:, 2]], dim=-1)

  def _heading_frame_reference(motion, times):
    mean_lin_vel = _centered_mean(motion.root_lin_vel_w, times, motion.motion_ends)
    mean_yaw_rate = _centered_mean(
      motion.root_ang_vel_w[:, 2], times, motion.motion_ends
    )
    lin_vel = quat_apply_inverse(yaw_quat(motion.root_quat_w[times]), mean_lin_vel)
    return torch.cat([lin_vel[:, :2], mean_yaw_rate[:, None]], dim=-1)

  def extract(motion, motion_times, aux_state):
    if mode == "follow":
      return _heading_frame_reference(motion, motion_times)
    cmd = _body_frame_cmd(motion, motion_times, aux_state[:, 3:7])
    noise = (torch.rand_like(cmd) * 2 - 1) * noise_half.to(cmd.device)
    return _clamp(cmd + noise)

  def precompute_replay_goal(motion):
    """Return the noise-free commands used for reference replay."""
    device = motion.root_lin_vel_w.device
    frame_count = motion.root_lin_vel_w.shape[0]
    if mode == "follow":
      # Limit the temporary [frames, window] indexing buffers for large libraries.
      commands = torch.empty_like(motion.root_lin_vel_w)
      for start in range(0, frame_count, 65_536):
        stop = min(start + 65_536, frame_count)
        times = torch.arange(start, stop, device=device)
        commands[start:stop] = _heading_frame_reference(motion, times)
      return commands
    times = torch.arange(frame_count, device=device)
    cmd = _body_frame_cmd(motion, times, motion.root_quat_w)
    return _clamp(cmd)

  def slider(name: str, bounds: tuple[float, float]) -> ContinuousGoalSpec:
    lower, upper = bounds
    default = min(upper, max(lower, 0.0)) if mode == "follow" else 0.0
    return ContinuousGoalSpec(name, lower, upper, step=0.1, default=default)

  return TaskSpec(
    name="velocity",
    goal_dim=3,
    extract_training_goal=extract,
    goal_specs=(
      slider("Lin Vel X (m/s)", lin_vel_x_range),
      slider("Lin Vel Y (m/s)", lin_vel_y_range),
      slider("Ang Vel Z (rad/s)", ang_vel_z_range),
    ),
    build_inference_goal_obs=lambda raw: raw,
    raw_goal_dim=3,
    precompute_replay_goal=precompute_replay_goal,
  )


@dataclass(frozen=True)
class VelocityTaskCfg:
  """Velocity-command task with optional explicit speed following."""

  lin_vel_x_range: tuple[float, float] = (-1.5, 1.5)
  lin_vel_y_range: tuple[float, float] = (-0.5, 0.5)
  ang_vel_z_range: tuple[float, float] = (-3.14, 3.14)
  window_frames: int = 25
  """Centered-mean window over motion-velocity command. 0/1 = raw instantaneous."""
  command_noise: tuple[float, float, float] = (0.5, 0.5, 0.785)
  """Uniform noise half-width per axis (lin_x, lin_y, ang_z) added to the
  velocity command in imitation mode."""
  mode: Literal["imitation", "follow"] = "imitation"
  reference_probability: float = 0.5
  """Probability of drawing a reference command source at each renewal."""
  standing_probability: float = 0.1
  """Probability of a zero command conditional on the independent source."""
  hold_seconds: float = 5.0
  """Time between renewals of the command source and independent command."""
  command_bins: tuple[int, int, int] = (5, 5, 5)
  """Number of equal-width sampling bins in vx, vy and yaw rate."""

  def __post_init__(self) -> None:
    if self.mode != "follow":
      return
    if not 0.0 <= self.reference_probability <= 1.0:
      raise ValueError("reference_probability must lie in [0, 1]")
    if not 0.0 <= self.standing_probability <= 1.0:
      raise ValueError("standing_probability must lie in [0, 1]")
    if not isfinite(self.hold_seconds) or self.hold_seconds <= 0.0:
      raise ValueError("hold_seconds must be finite and positive")
    if len(self.command_bins) != 3 or any(
      not isinstance(n, int) or n <= 0 for n in self.command_bins
    ):
      raise ValueError("command_bins must contain three positive counts")
    ranges = (self.lin_vel_x_range, self.lin_vel_y_range, self.ang_vel_z_range)
    if any(not (isfinite(lo) and isfinite(hi) and lo <= hi) for lo, hi in ranges):
      raise ValueError("velocity ranges must have finite ordered endpoints")


@dataclass(frozen=True)
class TasksCfg:
  """Per-task hyperparameters. Field order defines slot block ordering."""

  velocity: VelocityTaskCfg = field(default_factory=VelocityTaskCfg)


def resolve_task_specs(
  tasks: TasksCfg,
  representation: "StateRepresentation",
  num_goals: int,
  device: str = "cpu",
) -> list[TaskSpec]:
  return [
    make_motion_task(representation, num_goals),
    make_velocity_task(
      tasks.velocity.lin_vel_x_range,
      tasks.velocity.lin_vel_y_range,
      tasks.velocity.ang_vel_z_range,
      tasks.velocity.window_frames,
      tasks.velocity.command_noise,
      device,
      mode=tasks.velocity.mode,
    ),
  ]
