"""World-model training for standalone G1 velocity control."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch
import wandb
from tensordict import TensorDict
from torch import nn

from supermimic.envs.velocity_world_model_env import VelocityWorldModelEnv
from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
)
from supermimic.models import _CudaGraphNormalization, build_mlp
from supermimic.replay_buffer import FieldSpec
from supermimic.representations import GenCoordRepresentation, StateRepresentation
from supermimic.runners.supermimic_runner import (
  WorldModel,
  WorldModelRunner,
  WorldModelRunnerCfg,
  build_world_model,
)
from supermimic.tasks.velocity_env_cfg import VelocityCommand

VELOCITY_TASK_ID = "SuperMimic-GenCoord-G1-Velocity"


@dataclass
class VelocityRunnerCfg(WorldModelRunnerCfg):
  actor_obs: str = "proprio"
  actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
  critic_hidden_dims: tuple[int, ...] = (256, 128)
  policy: ActorCriticPolicyLearnerCfg = field(
    default_factory=ActorCriticPolicyLearnerCfg
  )
  warmup_wm_updates: int = 16
  """Dynamics updates on standing and exploratory trajectories before policy training."""
  exploration_noise: float = 0.15
  """Bounded, temporally smoothed noise on raw actions during real collection."""
  action_limit: float = 3.0

  def __post_init__(self) -> None:
    if self.representation != "gencoord" or self.actor_obs != "proprio":
      raise ValueError(
        "The velocity task uses gencoord dynamics and proprio actor input"
      )
    if self.wm.horizon < 2 or self.wm.horizon > self.buffer_size:
      raise ValueError("The world model horizon must be between 2 and buffer_size")
    if self.history_length < 1 or self.history_length > self.buffer_size:
      raise ValueError("The history length must be between 1 and buffer_size")
    if self.warmup_wm_updates < 0:
      raise ValueError("warmup_wm_updates must be non-negative")
    if self.policy.batch_size < 4 or self.policy.batch_size % 4:
      raise ValueError("The policy batch size must be a positive multiple of four")
    if not math.isfinite(self.action_limit) or self.action_limit <= 0:
      raise ValueError("action_limit must be finite and positive")
    if (
      not math.isfinite(self.exploration_noise)
      or not 0 <= self.exploration_noise <= self.action_limit
    ):
      raise ValueError("exploration_noise must be finite and within the action limit")


class VelocityActor(nn.Module):
  """A single policy head for proprioception and the live three-axis command."""

  def __init__(self, input_dim: int, num_actions: int, cfg: VelocityRunnerCfg) -> None:
    super().__init__()
    self.normalizer = _CudaGraphNormalization(input_dim)
    self.network = build_mlp(input_dim, cfg.actor_hidden_dims, output_dim=num_actions)
    self.action_limit = cfg.action_limit
    # Start close to the default joint targets when the actor takes over.
    with torch.no_grad():
      self.network[-1].weight.mul_(0.01)
      self.network[-1].bias.zero_()

  def forward(self, obs: TensorDict) -> torch.Tensor:
    return self.action_limit * self.network(self.normalizer(obs["actor"])).tanh()


class VelocityRunner(WorldModelRunner[VelocityRunnerCfg]):
  skip_empty_replay_windows = True

  @property
  def policy_warmup_updates(self) -> int:
    return self.cfg.warmup_wm_updates

  def _build_world_model(self) -> WorldModel:
    # The velocity objective uses pelvis and joint states directly. Bind the
    # robot's limits without the motion clip's forward-kinematics provider.
    return build_world_model(
      self.cfg, self.env, self.device, bind_representation=StateRepresentation.bind
    )

  def _bind_task(self) -> None:
    term = self.env.unwrapped.command_manager.get_term("twist")
    if not isinstance(term, VelocityCommand):
      raise TypeError("The velocity runner requires a VelocityCommand term named twist")
    self.velocity_command = term

  def _task_buffer_schema(self) -> dict[str, FieldSpec]:
    return {"velocity_command": FieldSpec(3), "command_time_left": FieldSpec()}

  def _transition_task_fields(self) -> dict[str, torch.Tensor]:
    return {
      "velocity_command": self.velocity_command.command.clone(),
      "command_time_left": self.velocity_command.time_left.clone(),
    }

  def _init_policy(self) -> None:
    representation = cast(GenCoordRepresentation, self._representation)
    sensor_dim = representation.proprio_obs_dim(self._num_actions)
    actor_dim = self.cfg.history_length * sensor_dim + self._num_goals
    critic_dim = (
      representation.rep_state_dim + self._num_goals + self._sim_context.output_dim
    )
    actor = VelocityActor(actor_dim, self._num_actions, self.cfg).to(self.device)
    critic = build_mlp(critic_dim, self.cfg.critic_hidden_dims, output_dim=1).to(
      self.device
    )

    self.wm_env = VelocityWorldModelEnv(
      dynamics=self._dynamics,
      representation=representation,
      env=self.env,
      batch_size=self.cfg.policy.batch_size,
      replay_buffer=self.replay_buffer,
      history_length=self.cfg.history_length,
      termination_softness=self.cfg.termination_softness,
      device=self.device,
    )
    if torch.device(self.device).type == "cuda":
      self.wm_env.reset = torch.compile(self.wm_env.reset)
      self.wm_env.step = torch.compile(self.wm_env.step)
    self.policy_learner = ActorCriticPolicyLearner(
      actor, critic, self.wm_env, self.cfg.policy, device=self.device
    )
    self._action_noise = torch.zeros(
      self.env.num_envs, self._num_actions, device=self.device
    )

  def _update_actor_normalization(self, actor_inputs: tuple[TensorDict, ...]) -> None:
    actor = cast(VelocityActor, self.policy_learner.actor)
    actor.normalizer.train()
    actor.normalizer.update(torch.cat([obs["actor"] for obs in actor_inputs]))

  def step(self) -> int:
    with torch.inference_mode():
      obs = self.env.get_observations()
      state = obs["full_state"].to(self.device)
      sim_actor_obs = obs["actor"].to(self.device)
      command = self.velocity_command.command.clone()
      sim_ctx = self._sim_context.read(self.env.unwrapped)
      sensor = self._representation.add_proprio_noise(sim_actor_obs)
      actor_obs = self._representation.build_actor_obs(
        self._collection_history(sensor), command
      )
      actor_td = TensorDict({"actor": actor_obs}, [self.env.num_envs])

      self._action_noise.lerp_(
        (2.0 * torch.rand_like(self._action_noise) - 1.0) * self.cfg.exploration_noise,
        0.3,
      )
      actions = self._action_noise
      if self._num_wm_updates >= self.policy_warmup_updates:
        actions = self.policy_learner.actor(actor_td) + actions
      actions = actions.clamp(-self.cfg.action_limit, self.cfg.action_limit)

      extra_fields = {"sim_ctx": sim_ctx}
      if self._stores_actor_obs:
        extra_fields["actor_obs"] = sim_actor_obs
      next_obs, rewards, dones, extras = self._step_and_store(
        actions, state, actor_td=actor_td, extra_fields=extra_fields
      )
      self._action_noise.masked_fill_(dones.bool().unsqueeze(-1), 0.0)
      self._advance_sim_history(next_obs["actor"].to(self.device), dones)
      terminated = self.env.unwrapped.termination_manager.terminated
      self.logger.process_env_step(rewards, terminated, extras)

    self._flush_normalization()
    return self.env.num_envs

  def _save_checkpoint(self, checkpoint_dir: Path) -> list[str]:
    with self._stage_checkpoint(checkpoint_dir) as staging_dir:
      torch.save(self._wm.bundle(), staging_dir / "world_model.pt")
      torch.save(self.policy_learner.actor.state_dict(), staging_dir / "actor.pt")
      torch.save(self.policy_learner.critic.state_dict(), staging_dir / "critic.pt")
    return [
      str(checkpoint_dir / name)
      for name in ("world_model.pt", "actor.pt", "critic.pt", "runner_config.json")
    ]

  def _publish_checkpoint(self, files: list[str]) -> None:
    if wandb.run is not None and wandb.run.settings.mode == "online":
      super()._publish_checkpoint(files)
