"""Differentiable G1 velocity-control rollouts on the learned world model."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

from supermimic.envs.world_model_env import DifferentiableWorldModelEnv
from supermimic.tasks.velocity_env_cfg import advance_velocity_commands
from supermimic.utils.velocity_reward import velocity_fall_error, velocity_reward

if TYPE_CHECKING:
  from supermimic.models import Dynamics
  from supermimic.replay_buffer import ReplayBuffer
  from supermimic.representations import GenCoordRepresentation


class VelocityWorldModelEnv(DifferentiableWorldModelEnv):
  """Use sampled body-frame commands to score imagined G1 states."""

  def __init__(
    self,
    dynamics: Dynamics,
    representation: GenCoordRepresentation,
    env: Any,
    batch_size: int,
    replay_buffer: ReplayBuffer,
    history_length: int,
    termination_softness: float,
    device: str = "cpu",
  ) -> None:
    super().__init__(
      dynamics=dynamics,
      representation=representation,
      env=env,
      batch_size=batch_size,
      replay_buffer=replay_buffer,
      history_length=history_length,
      termination_softness=termination_softness,
      device=device,
    )
    self._command_cfg = env.cfg.commands["twist"]
    reward_term = env.cfg.rewards["velocity"]
    self._reward_cfg = reward_term.params["cfg"]
    if not math.isfinite(reward_term.weight) or reward_term.weight < 0.0:
      raise ValueError("The velocity reward term weight must be finite and nonnegative")
    reward_dt = representation.dt if env.cfg.scale_rewards_by_dt else 1.0
    self._step_reward_scale = reward_term.weight * reward_dt
    self._fall_cfg = env.cfg.terminations["fall"].params["cfg"]
    self._joint_limits = env.unwrapped.scene["robot"].data.soft_joint_pos_limits[0]
    self.max_step_reward = self._reward_cfg.max_reward * self._step_reward_scale

  def _on_reset(self, batch: TensorDict) -> None:
    # The newest replay row contains the command that produced its action.
    self.velocity_command = batch["velocity_command"]
    self.command_time_left = batch["command_time_left"]
    self._reset_sensor_history(batch)

  def _compute_commands(self) -> None:
    self.velocity_command, self.command_time_left = advance_velocity_commands(
      self.velocity_command,
      self.command_time_left,
      dt=self.representation.dt,
      cfg=self._command_cfg,
    )

  def get_observations(self) -> TensorDict:
    actor = self.representation.build_actor_obs(
      self._sensor_history(), self.velocity_command
    )
    critic = torch.cat([self.rep_state, self.velocity_command, self.sim_ctx], dim=-1)
    return TensorDict(
      {"actor": actor, "critic": critic}, [self.num_envs], device=self.device
    )

  def _wm_step(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    next_aux_state, next_rep_state = self.dynamics(
      rep_state, action, aux_state, sim_ctx
    )
    next_joint_state = self.representation.joint_state(next_rep_state)
    fall_error = velocity_fall_error(next_rep_state, self._fall_cfg)
    term_prob = torch.sigmoid((fall_error - 1.0) / self._termination_softness)

    # The current command scores the transition, even when the next observation
    # gets a resampled command. mjlab applies the same dt factor in RewardManager.
    reward = self._step_reward_scale * velocity_reward(
      next_rep_state,
      self.velocity_command,
      action,
      last_action,
      self._joint_limits,
      self._reward_cfg,
    )
    return next_aux_state, next_rep_state, next_joint_state, term_prob, reward
