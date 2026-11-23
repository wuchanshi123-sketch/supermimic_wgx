"""Differentiable velocity-control rollouts on the learned world model."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

from supermimic.envs.world_model_env import DifferentiableWorldModelEnv
from supermimic.learners.amp_style import amp_command_gate
from supermimic.tasks.velocity_env_cfg import (
  SimVelocityReward,
  advance_velocity_commands,
)
from supermimic.utils.gencoord_helpers import decompose_rep_state
from supermimic.utils.velocity_angmom import VelocityAngularMomentum
from supermimic.utils.velocity_reward import velocity_fall_error
from supermimic.utils.velocity_torso import VelocityTorsoKinematics

if TYPE_CHECKING:
  from supermimic.learners.amp_style import AmpDiscriminator, AmpStyleCfg
  from supermimic.models import Dynamics
  from supermimic.replay_buffer import ReplayBuffer
  from supermimic.representations import GenCoordRepresentation
  from supermimic.sim_context import SimContext
  from supermimic.utils.amp_body_features import AmpBodyFeatures
  from supermimic.utils.velocity_collision import VelocityRewardSignalModel


class VelocityWorldModelEnv(DifferentiableWorldModelEnv):
  """Use sampled body-frame commands to score imagined robot states."""

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
    *,
    sim_context: SimContext,
    reward_signals: VelocityRewardSignalModel,
    amp_discriminator: AmpDiscriminator | None = None,
    amp_cfg: AmpStyleCfg | None = None,
    amp_features: AmpBodyFeatures | None = None,
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
    reward_term = env.unwrapped.reward_manager.get_term_cfg("velocity")
    self._reward_cfg = reward_term.params["cfg"]
    self._evaluate_reward = reward_term.func.evaluate
    torso_name = getattr(
      reward_term.func, "torso_body_name", SimVelocityReward.torso_body_name
    )
    self._torso = VelocityTorsoKinematics(env.unwrapped, torso_name)
    self._angular_momentum = None
    if amp_features is not None:
      self._angular_momentum = amp_features.kinematics
    elif self._reward_cfg.angular_momentum_cost > 0.0:
      self._angular_momentum = VelocityAngularMomentum(env.unwrapped)
    self._sim_context = sim_context
    self._com_slice = sim_context.find("base_com")
    self._reward_signals = reward_signals
    if not math.isfinite(reward_term.weight) or reward_term.weight < 0.0:
      raise ValueError("The velocity reward term weight must be finite and nonnegative")
    reward_dt = representation.dt if env.cfg.scale_rewards_by_dt else 1.0
    self._step_reward_scale = reward_term.weight * reward_dt
    amp_enabled = amp_features is not None
    if amp_enabled != (amp_discriminator is not None) or amp_enabled != (
      amp_cfg is not None
    ):
      raise ValueError("AMP reward requires a discriminator, features and a config")
    self._amp_discriminator = amp_discriminator
    self._amp_features = amp_features
    self._amp_step_scale = amp_cfg.weight * reward_dt if amp_cfg is not None else 0.0
    self._style_command_gate = (
      amp_cfg.style_command_gate if amp_cfg is not None else 0.0
    )
    self._style_speed_gate = amp_cfg is not None and amp_cfg.style_speed_gate
    self._fall_cfg = env.cfg.terminations["fall"].params["cfg"]
    self._joint_limits = env.unwrapped.scene["robot"].data.soft_joint_pos_limits[0]
    self.max_step_reward = (
      self._reward_cfg.max_reward * self._step_reward_scale + self._amp_step_scale
    )

  def _on_reset(self, batch: TensorDict) -> None:
    # The newest replay row contains the command that produced its action.
    self.velocity_command = batch["velocity_command"]
    self.command_time_left = batch["command_time_left"]
    self._reset_sensor_history(batch)
    if self._amp_features is not None:
      self._amp_previous_features = self._amp_features(self.rep_state)

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
    torso_tilt_error, torso_roll_pitch_speed_sq = self._torso(
      next_rep_state, next_aux_state[..., 3:7]
    )
    kin = None
    if self._amp_features is not None:
      assert self._angular_momentum is not None
      kin = self._angular_momentum.body_kinematics(next_rep_state)
      next_features = self._amp_features(next_rep_state, kin)
      assert self._amp_discriminator is not None
      style = self._amp_discriminator.reward(self._amp_previous_features, next_features)
      self._amp_previous_features = next_features

    angular_momentum_l2 = torch.zeros_like(term_prob)
    if self._reward_cfg.angular_momentum_cost > 0.0:
      assert self._angular_momentum is not None
      com_ipos = (
        self._sim_context.denormalize(sim_ctx, self._com_slice)
        if self._com_slice is not None
        else None
      )
      angular_momentum_l2 = (
        self._angular_momentum(next_rep_state, com_ipos)
        if kin is None
        else self._angular_momentum.from_kinematics(kin, com_ipos)
      )
    joint_acc_l2, collision_count, foot_slip, soft_landing = self._reward_signals(
      rep_state, action, sim_ctx
    ).unbind(-1)

    # The current command scores the transition, even when the next observation
    # gets a resampled command. mjlab applies the same dt factor in RewardManager.
    reward = self._step_reward_scale * self._evaluate_reward(
      next_rep_state,
      self.velocity_command,
      action,
      last_action,
      self._joint_limits,
      self._reward_cfg,
      torso_tilt_error=torso_tilt_error,
      torso_roll_pitch_speed_sq=torso_roll_pitch_speed_sq,
      joint_acc_l2=joint_acc_l2,
      angular_momentum_l2=angular_momentum_l2,
      self_collision_count=collision_count,
      foot_slip=foot_slip,
      soft_landing=soft_landing,
      terminated=term_prob,
    )
    if self._amp_features is not None:
      gate = amp_command_gate(self.velocity_command, self._style_command_gate)
      style_bonus = self._amp_step_scale * gate * style
      if self._style_speed_gate:
        _, _, lin_vel, ang_vel, _, _ = decompose_rep_state(next_rep_state)
        xy_error = (lin_vel[..., :2] - self.velocity_command[..., :2]).square().sum(-1)
        yaw_error = (ang_vel[..., 2] - self.velocity_command[..., 2]).square()
        tracking = torch.exp(
          -xy_error / self._reward_cfg.linear_std**2
          - yaw_error / self._reward_cfg.yaw_std**2
        )
        style_bonus = style_bonus * tracking
      reward = reward + style_bonus
    return next_aux_state, next_rep_state, next_joint_state, term_prob, reward
