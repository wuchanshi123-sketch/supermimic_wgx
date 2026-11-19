"""Configure the MuJoCo slots for the velocity following task."""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import wraps
from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import matrix_from_quat

from supermimic.utils.velocity_reward import (
  heading_velocity,
  velocity_fallen,
  velocity_follow_reward,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


@dataclass(kw_only=True)
class VelocityFollowMotionCfg(MotionCommandCfg):
  velocity_start: int

  def build(self, env: ManagerBasedRlEnv) -> VelocityFollowMotion:
    return VelocityFollowMotion(self, env)


class VelocityFollowMotion(MotionCommand):
  """Keep the reference cursor inside its clip without resetting a velocity slot."""

  def __init__(self, cfg: VelocityFollowMotionCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.velocity_start = cfg.velocity_start
    self.velocity_goal = torch.zeros(
      env.num_envs - self.velocity_start, 3, device=env.device
    )
    self.velocity_measured = torch.zeros_like(self.velocity_goal)
    self.torso_index = self.robot.body_names.index("torso_link")

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    ids = (
      slice(self.velocity_start, None)
      if env_ids is None
      else env_ids[env_ids >= self.velocity_start]
    )
    ends = self.motion.motion_ends
    frames = self.time_steps[ids]
    clip_end = ends[torch.searchsorted(ends, frames, right=True)]
    # The base update increments the cursor. Compensate for that increment at
    # a velocity clip's last frame so it cannot wrap and teleport the robot.
    self.time_steps[ids] = torch.minimum(frames, clip_end - 2)
    super()._update_command(env_ids)


def _follow_motion(env: ManagerBasedRlEnv) -> VelocityFollowMotion:
  return cast(VelocityFollowMotion, env.command_manager.get_term("motion"))


def _only_motion_slots(func, velocity_start: int):
  @wraps(func)
  def masked(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
    value = func(env, **params)
    return torch.cat((value[:velocity_start], torch.zeros_like(value[velocity_start:])))

  return masked


def _follow_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  motion = _follow_motion(env)
  start = motion.velocity_start
  slots = slice(start, None)
  data = motion.robot.data
  torso = motion.torso_index
  quat = data.root_link_quat_w[slots]
  lin_vel = data.root_link_lin_vel_w[slots]
  ang_vel = data.root_link_ang_vel_w[slots]
  motion.velocity_measured = heading_velocity(quat, lin_vel, ang_vel)
  # RewardManager applies the control-period scaling.
  reward = velocity_follow_reward(
    motion.velocity_goal,
    quat,
    lin_vel,
    ang_vel,
    matrix_from_quat(data.body_link_quat_w[slots, torso]),
    data.body_link_ang_vel_w[slots, torso],
    data.joint_pos[slots],
    data.soft_joint_pos_limits[slots],
    env.action_manager.action[slots],
    env.action_manager.prev_action[slots],
    dt=1.0,
  )
  return torch.cat((reward.new_zeros(start), reward))


def _follow_fall(env: ManagerBasedRlEnv) -> torch.Tensor:
  motion = _follow_motion(env)
  start = motion.velocity_start
  data = motion.robot.data
  root_pos = data.root_link_pos_w[start:] - env.scene.env_origins[start:]
  fallen = velocity_fallen(root_pos, data.root_link_quat_w[start:])
  return torch.cat((fallen.new_zeros(start), fallen))


def configure_velocity_follow(
  env_cfg: ManagerBasedRlEnvCfg, velocity_start: int
) -> None:
  """Assign reference rewards to motion slots and velocity rewards to follow slots."""
  if not isinstance(env_cfg.commands.get("motion"), MotionCommandCfg):
    raise ValueError("Velocity following requires a motion reference command")
  motion = env_cfg.commands["motion"]
  options = {
    field.name: getattr(motion, field.name) for field in fields(MotionCommandCfg)
  }
  if min(motion.resampling_time_range) <= env_cfg.episode_length_s:
    raise ValueError(
      "Velocity following needs the reference reset timer longer than the episode"
    )
  env_cfg.commands["motion"] = VelocityFollowMotionCfg(
    **options, velocity_start=velocity_start
  )
  for term in env_cfg.rewards.values():
    if term is not None:
      term.func = _only_motion_slots(term.func, velocity_start)
  env_cfg.rewards["velocity_follow"] = RewardTermCfg(func=_follow_reward, weight=1.0)
  for name, term in env_cfg.terminations.items():
    if term is not None and name != "time_out":
      term.func = _only_motion_slots(term.func, velocity_start)
  env_cfg.terminations["velocity_fall"] = TerminationTermCfg(func=_follow_fall)
