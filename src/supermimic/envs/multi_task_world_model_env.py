"""Multi-task differentiable world model env for concurrent encoder training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.tasks.velocity_commands import VelocityCommands, reference_bc_mask
from supermimic.utils.velocity_reward import (
  VELOCITY_MAX_REWARD_RATE,
  velocity_fall_probability,
  velocity_follow_reward,
)

if TYPE_CHECKING:
  from supermimic.models import Dynamics
  from supermimic.replay_buffer import ReplayBuffer
  from supermimic.representations import StateRepresentation
  from supermimic.tasks.task_spec import TaskSpec, VelocityTaskCfg


class MultiTaskWorldModelEnv(WorldModelEnv):
  """WM env for motion tracking and opt-in velocity following.

  Observations emit per-task goals over the full batch and the actor slices
  its own slot range. Velocity slots use their own reward and fall criteria
  in follow mode. The observation history lives in the base class.
  """

  def __init__(
    self,
    dynamics: Dynamics,
    representation: StateRepresentation,
    env: Any,
    batch_size: int,
    replay_buffer: ReplayBuffer,
    task_specs: list[TaskSpec],
    history_length: int,
    termination_softness: float,
    device: str = "cpu",
    *,
    velocity_cfg: VelocityTaskCfg | None = None,
    velocity_start: int | None = None,
    sim_velocity_start: int | None = None,
  ) -> None:
    self.task_specs = task_specs
    assert task_specs[0].name == "motion", "task_specs[0] must be the motion task"
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
    self._velocity_commands: VelocityCommands | None = None
    self.velocity_max_step_reward: float | None = None
    if velocity_cfg is not None:
      if velocity_start is None or not 0 < velocity_start < batch_size:
        raise ValueError("velocity follow requires motion and velocity imagined slots")
      if sim_velocity_start is None or not 0 < sim_velocity_start < env.num_envs:
        raise ValueError("velocity follow requires motion and velocity simulator slots")
      velocity_spec = next(
        (task for task in task_specs if task.name == "velocity"), None
      )
      if velocity_spec is None or velocity_spec.precompute_replay_goal is None:
        raise ValueError("velocity follow requires reference velocity goals")

      self._velocity_start = velocity_start
      self._sim_velocity_start = sim_velocity_start
      self._velocity_slots = slice(velocity_start, None)
      self._velocity_dt = env.unwrapped.step_dt
      self.velocity_max_step_reward = VELOCITY_MAX_REWARD_RATE * self._velocity_dt
      self._joint_limits = env.unwrapped.scene["robot"].data.soft_joint_pos_limits[0]
      motion_cmd = env.unwrapped.command_manager.get_term("motion")
      self._root_idx = motion_cmd.cfg.body_names.index("pelvis")
      self._torso_idx = motion_cmd.cfg.body_names.index("torso_link")
      self._velocity_commands = VelocityCommands(
        velocity_cfg,
        velocity_spec.precompute_replay_goal(self.motion),
        batch_size - velocity_start,
        self._velocity_dt,
        device,
      )
      self._is_velocity = torch.arange(batch_size, device=device) >= velocity_start

  def sample_initial_states(self) -> TensorDict:
    if self._velocity_commands is None:
      return super().sample_initial_states()
    # Preserve correspondence between simulator and imagined task slots, even
    # when their batch sizes (and therefore slice boundaries) are different.
    motion = self._sample_initial_batch(
      self._velocity_start, env_range=(0, self._sim_velocity_start)
    )
    velocity = self._sample_initial_batch(
      self.num_envs - self._velocity_start,
      env_range=(self._sim_velocity_start, self.replay_buffer.num_envs),
    )
    return torch.cat([motion, velocity], dim=0)

  def _on_reset(self, batch: TensorDict) -> None:
    if self._velocity_commands is not None:
      slots = self._velocity_slots
      self._command_state = (
        batch["velocity_command"][slots],
        batch["velocity_reference"][slots],
        batch["velocity_steps"][slots],
      )
    super()._on_reset(batch)

  def _compute_commands(self) -> None:
    # The base step scores the current action before refreshing its command.
    if self._velocity_commands is not None:
      self._command_state = self._velocity_commands.advance(*self._command_state)
    super()._compute_commands()

  def _build_task_goals(self) -> None:
    # The anchor task's goal is the actor goal already built in _update_goal; reuse it.
    anchor, *rest = self.task_specs
    self._task_goal_obs: dict[str, torch.Tensor] = {anchor.name: self._actor_goal_obs}
    for ts in rest:
      if ts.name == "velocity" and self._velocity_commands is not None:
        command, reference, _ = self._command_state
        slots = self._velocity_slots
        self._velocity_goal = self._velocity_commands.goal(
          command, reference, self.motion_times[slots]
        )
        self._task_goal_obs[ts.name] = torch.cat(
          [
            self._velocity_goal.new_zeros((self._velocity_start, 3)),
            self._velocity_goal,
          ]
        )
      else:
        self._task_goal_obs[ts.name] = ts.extract_training_goal(
          self.motion, self.motion_times, self.aux_state
        ).clone()

  def get_observations(self) -> TensorDict:
    critic = self._critic_obs()
    if self._velocity_commands is not None:
      critic = torch.cat(
        [
          critic,
          self._task_goal_obs["velocity"],
          self._is_velocity[:, None].to(critic),
        ],
        dim=-1,
      )
    obs = build_actor_obs_td(self._task_goal_obs, self._sensor_history(), critic=critic)
    if self._velocity_commands is not None:
      _, reference, _ = self._command_state
      bc_mask = reference_bc_mask(
        reference, self.motion_times[self._velocity_slots], self.motion.motion_ends
      )
      obs["velocity_reference"] = torch.cat(
        [bc_mask.new_zeros((self._velocity_start,)), bc_mask]
      )
    return obs

  def _score_transition(
    self,
    next_rep_state: torch.Tensor,
    next_aux_state: torch.Tensor,
    next_joint_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
  ) -> tuple[torch.Tensor, torch.Tensor]:
    legacy = super()._score_transition
    if self._velocity_commands is None:
      return legacy(
        next_rep_state,
        next_aux_state,
        next_joint_state,
        action,
        last_action,
        reference,
      )
    motion = slice(None, self._velocity_start)
    velocity = self._velocity_slots
    pos, quat, lin, ang = reference
    motion_term, motion_reward = legacy(
      next_rep_state[motion],
      next_aux_state[motion],
      next_joint_state[motion],
      action[motion],
      last_action[motion],
      (pos[motion], quat[motion], lin[motion], ang[motion]),
    )
    aux = next_aux_state[velocity]
    _, body_rot_w, body_lin_vel_w, body_ang_vel_w = (
      self.representation.world_frame_bodies(aux, next_rep_state[velocity])
    )
    joint_pos, _ = next_joint_state[velocity].chunk(2, dim=-1)
    velocity_reward = velocity_follow_reward(
      self._velocity_goal,
      aux[:, 3:7],
      body_lin_vel_w[:, self._root_idx],
      body_ang_vel_w[:, self._root_idx],
      body_rot_w[:, self._torso_idx],
      body_ang_vel_w[:, self._torso_idx],
      joint_pos,
      self._joint_limits,
      action[velocity],
      last_action[velocity],
      self._velocity_dt,
    )
    velocity_term = velocity_fall_probability(
      aux[:, :3], aux[:, 3:7], self._termination_softness
    )
    return (
      torch.cat([motion_term, velocity_term]),
      torch.cat([motion_reward, velocity_reward]),
    )
