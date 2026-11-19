"""G1 flat-ground velocity commands, observations, and rewards."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers import (
  CommandTerm,
  CommandTermCfg,
  MetricsTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from supermimic.tasks.env_cfgs import (
  _sensor_actor_group,
  joint_pos,
  joint_vel,
  root_ang_vel,
  root_lin_vel,
  root_ori,
  root_pos,
)
from supermimic.utils.velocity_reward import (
  VelocityRewardCfg,
  VelocityTerminationCfg,
  velocity_fall_error,
  velocity_reward,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


@dataclass(kw_only=True)
class VelocityCommandCfg(CommandTermCfg):
  resampling_time_range: tuple[float, float] = (3.0, 8.0)
  lin_vel_x: tuple[float, float] = (-1.0, 1.0)
  lin_vel_y: tuple[float, float] = (-1.0, 1.0)
  ang_vel_z: tuple[float, float] = (-3.0, 3.0)
  standing_fraction: float = 0.10
  single_axis_fraction: float = 0.45
  external: bool = False
  """Hold an externally supplied command until reset instead of sampling."""

  def __post_init__(self) -> None:
    for limits in (self.lin_vel_x, self.lin_vel_y, self.ang_vel_z):
      if not (
        math.isfinite(limits[0]) and math.isfinite(limits[1]) and limits[0] <= limits[1]
      ):
        raise ValueError("Velocity command ranges must have finite, ascending limits")
    min_time, max_time = self.resampling_time_range
    if not (0.0 < min_time <= max_time and math.isfinite(max_time)):
      raise ValueError("Velocity command hold times must be positive and ascending")
    if not (
      self.standing_fraction >= 0
      and self.single_axis_fraction >= 0
      and self.standing_fraction + self.single_axis_fraction <= 1
    ):
      raise ValueError(
        "Standing and single-axis command fractions must sum to at most one"
      )

  def build(self, env: ManagerBasedRlEnv) -> VelocityCommand:
    return VelocityCommand(self, env)


def _sample_velocity_command(
  like: torch.Tensor, cfg: VelocityCommandCfg
) -> torch.Tensor:
  """Sample standing, single-axis, and combined commands at a fixed batch shape."""
  axes = (cfg.lin_vel_x, cfg.lin_vel_y, cfg.ang_vel_z)
  lower = like.new_tensor(tuple(axis[0] for axis in axes))
  upper = like.new_tensor(tuple(axis[1] for axis in axes))
  command = lower + torch.rand_like(like) * (upper - lower)

  mode = torch.rand_like(like[..., :1])
  axis = torch.randint(3, mode.shape, device=like.device)
  axis_mask = axis == torch.arange(3, device=like.device)
  standing = mode < cfg.standing_fraction
  combined = mode >= cfg.standing_fraction + cfg.single_axis_fraction
  return torch.where(~standing & (combined | axis_mask), command, 0.0)


def advance_velocity_commands(
  command: torch.Tensor,
  time_left: torch.Tensor,
  *,
  dt: float,
  cfg: VelocityCommandCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Advance command holds after scoring a step, keeping imagined shapes static."""
  remaining = time_left - dt
  expired = remaining <= 0.0
  min_time, max_time = cfg.resampling_time_range
  new_duration = min_time + torch.rand_like(time_left) * (max_time - min_time)
  new_command = _sample_velocity_command(command, cfg)
  return (
    torch.where(expired.unsqueeze(-1), new_command, command),
    torch.where(expired, new_duration, remaining),
  )


class VelocityCommand(CommandTerm):
  """Sample direct pelvis-frame commands and retain mjlab's per-env timers."""

  cfg: VelocityCommandCfg

  def __init__(self, cfg: VelocityCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self._command = torch.zeros((self.num_envs, 3), device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.cfg.external:
      self._command[env_ids] = 0.0
      self.time_left[env_ids] = torch.inf
    else:
      self._command[env_ids] = _sample_velocity_command(
        self._command[env_ids], self.cfg
      )

  def _update_metrics(self) -> None:
    pass

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    pass


def _velocity_command(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.command_manager.get_term("twist").command


def _pelvis_rep_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"].data
  return torch.cat(
    [
      root_pos(env)[..., 2:3],
      robot.projected_gravity_b,
      robot.root_link_lin_vel_b,
      robot.root_link_ang_vel_b,
      robot.joint_pos,
      robot.joint_vel,
    ],
    dim=-1,
  )


def sim_velocity_reward(env: ManagerBasedRlEnv, cfg: VelocityRewardCfg) -> torch.Tensor:
  """Evaluate the shared reward rate on MuJoCo's post-physics state."""
  return velocity_reward(
    _pelvis_rep_state(env),
    _velocity_command(env),
    env.action_manager.action,
    env.action_manager.prev_action,
    env.scene["robot"].data.soft_joint_pos_limits,
    cfg,
  )


def sim_velocity_fall(
  env: ManagerBasedRlEnv, cfg: VelocityTerminationCfg
) -> torch.Tensor:
  return velocity_fall_error(_pelvis_rep_state(env), cfg) >= 1.0


def _velocity_abs_error(env: ManagerBasedRlEnv, axis: int) -> torch.Tensor:
  robot = env.scene["robot"].data
  if axis == 2:
    actual = robot.root_link_ang_vel_b[..., 2]
  else:
    actual = robot.root_link_lin_vel_b[..., axis]
  return (_velocity_command(env)[..., axis] - actual).abs()


def velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the standalone task using the mjlab G1 flat-ground hardware setup."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.commands = {"twist": VelocityCommandCfg(external=play)}
  cfg.observations = {
    "actor": _sensor_actor_group(),
    "full_state": ObservationGroupCfg(
      terms={
        "root_pos": ObservationTermCfg(func=root_pos),
        "root_ori": ObservationTermCfg(func=root_ori),
        "joint_pos": ObservationTermCfg(func=joint_pos),
        "root_lin_vel": ObservationTermCfg(func=root_lin_vel),
        "root_ang_vel": ObservationTermCfg(func=root_ang_vel),
        "joint_vel": ObservationTermCfg(func=joint_vel),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "goal": ObservationGroupCfg(
      terms={"velocity": ObservationTermCfg(func=_velocity_command)},
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }
  cfg.rewards = {
    "velocity": RewardTermCfg(
      func=sim_velocity_reward, weight=1.0, params={"cfg": VelocityRewardCfg()}
    )
  }
  cfg.terminations = {
    "time_out": cfg.terminations["time_out"],
    "fall": TerminationTermCfg(
      func=sim_velocity_fall, params={"cfg": VelocityTerminationCfg()}
    ),
  }
  cfg.metrics = {
    f"{axis}_error": MetricsTermCfg(func=_velocity_abs_error, params={"axis": index})
    for index, axis in enumerate(("vx", "vy", "wz"))
  }
  cfg.events = {
    name: cfg.events[name]
    for name in ("reset_base", "reset_robot_joints", "foot_friction", "base_com")
  }
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.02, 0.02)
  cfg.curriculum = {}
  cfg.scene.sensors = ()
  return cfg
