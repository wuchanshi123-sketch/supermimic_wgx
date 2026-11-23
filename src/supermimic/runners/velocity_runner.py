"""World-model training for standalone velocity control."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch
import wandb
from tensordict import TensorDict
from torch import nn

from supermimic.envs.velocity_world_model_env import VelocityWorldModelEnv
from supermimic.learners.amp_style import AmpDiscriminator, AmpStyleCfg, AmpStyleLearner
from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
)
from supermimic.learners.velocity_wm_learner import (
  VelocityAmpTraining,
  VelocityWorldModelLearner,
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
from supermimic.tasks.velocity_env_cfg import SimVelocityReward, VelocityCommand
from supermimic.tasks.velocity_rsi import (
  REFERENCE_STATE_RESET_TERM,
  ReferenceStateReset,
)
from supermimic.utils.amp_body_features import (
  R00_AMP_ANCHOR_NAME,
  R00_AMP_BODY_NAMES,
  AmpBodyFeatures,
)
from supermimic.utils.amp_reference import AmpReference
from supermimic.utils.local_checkpoints import resolve_checkpoint_actor
from supermimic.utils.velocity_angmom import VelocityAngularMomentum
from supermimic.utils.velocity_collision import (
  VELOCITY_REWARD_SIGNAL_COUNT,
  VelocityRewardSignalModel,
)

VELOCITY_TASK_ID = "SuperMimic-GenCoord-G1-Velocity"
VELOCITY_AMP_TASK_ID = "SuperMimic-GenCoord-G1-Velocity-AMP"
R00_VELOCITY_TASK_ID = "SuperMimic-GenCoord-R00-Velocity"
R00_VELOCITY_AMP_TASK_ID = "SuperMimic-GenCoord-R00-Velocity-AMP"
R00_VELOCITY_TASK_IDS = (R00_VELOCITY_TASK_ID, R00_VELOCITY_AMP_TASK_ID)
VELOCITY_AMP_TASK_IDS = (VELOCITY_AMP_TASK_ID, R00_VELOCITY_AMP_TASK_ID)
VELOCITY_TASK_IDS = (VELOCITY_TASK_ID, VELOCITY_AMP_TASK_ID, *R00_VELOCITY_TASK_IDS)
AMP_DISCRIMINATOR_VERSION = 3


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
  init_checkpoint: str | None = None
  """Local checkpoint directory supplying actor.pt and world_model.pt for a new run."""
  amp: AmpStyleCfg = field(default_factory=AmpStyleCfg)

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
    if self.init_checkpoint is not None and self.wandb_run_path is not None:
      raise ValueError("Choose either init_checkpoint or wandb_run_path")
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
    self._init_actor_path = self._initial_actor_path()
    bundle = None
    # Bind the robot's limits without the motion clip's kinematics provider.
    wm = build_world_model(
      self.cfg, self.env, self.device, bind_representation=StateRepresentation.bind
    )
    self._reward_signals = VelocityRewardSignalModel(
      wm.representation.rep_state_dim, wm.num_actions, wm.sim_context.output_dim
    ).to(self.device)
    if self._init_actor_path is not None:
      model_path = self._init_actor_path.with_name("world_model.pt")
      bundle = torch.load(model_path, map_location="cpu", weights_only=True)
      try:
        for name in ("action_scale", "action_offset"):
          current = getattr(wm.dynamics, name).detach().cpu()
          if not torch.equal(bundle["dynamics"][name], current):
            raise ValueError(
              f"{model_path} uses a different {name} from the current env"
            )
        wm.load_bundle(bundle)
        if "velocity_reward_model" in bundle:
          self._reward_signals.load_checkpoint(bundle["velocity_reward_model"])
      except (KeyError, TypeError, RuntimeError) as exc:
        raise ValueError(
          f"World model from {model_path} does not match the current runner"
        ) from exc
    self._amp = self._initialize_amp(wm, bundle)
    return wm

  def _initialize_amp(
    self, wm: WorldModel, bundle: dict | None
  ) -> VelocityAmpTraining | None:
    cfg = self.cfg.amp
    if cfg.motion_file is None or cfg.weight == 0:
      return None
    env = self.env.unwrapped
    reference = None
    if REFERENCE_STATE_RESET_TERM in env.cfg.events:
      term = env.event_manager.get_term_cfg(REFERENCE_STATE_RESET_TERM)
      if (
        isinstance(term.func, ReferenceStateReset)
        and term.params["motion_file"] == cfg.motion_file
      ):
        reference = term.func.reference
    if reference is None:
      reference = AmpReference.load(
        cfg.motion_file, env.scene["robot"].joint_names, env.step_dt, self.device
      )
    kinematics = VelocityAngularMomentum(env)
    features = (
      AmpBodyFeatures(
        kinematics, body_names=R00_AMP_BODY_NAMES, anchor_name=R00_AMP_ANCHOR_NAME
      )
      if self.task_id in R00_VELOCITY_TASK_IDS
      else AmpBodyFeatures(kinematics)
    )
    discriminator = AmpDiscriminator(features.feature_dim, cfg).to(self.device)
    with torch.no_grad():
      expert_features = torch.cat(
        [
          features(wm.representation.represent(frames))
          for frames in reference.states.split(4096)
        ]
      )
      discriminator.fit_normalizer(expert_features)
    if bundle is not None and "amp_discriminator" in bundle:
      if bundle.get("amp_discriminator_version") == AMP_DISCRIMINATOR_VERSION:
        discriminator.load_state_dict(bundle["amp_discriminator"])
      else:
        print(
          "[INFO] Reinitializing the legacy AMP discriminator with torso-relative body features"
        )
    print(
      f"[INFO] AMP loaded {reference.num_clips} reference clips "
      f"({reference.num_transitions} transitions), reward weight {cfg.weight:g}"
    )
    return VelocityAmpTraining(
      AmpStyleLearner(discriminator, cfg, self.device),
      reference,
      features,
      expert_features,
    )

  def _make_world_model_learner(self) -> VelocityWorldModelLearner:
    return VelocityWorldModelLearner(
      self._dynamics,
      self._representation,
      self.cfg.wm,
      self.device,
      self._reward_signals,
      amp=self._amp,
    )

  def _initial_actor_path(self) -> Path | None:
    if self.cfg.init_checkpoint is None:
      return None
    directory = Path(self.cfg.init_checkpoint).expanduser()
    if not directory.is_dir():
      raise FileNotFoundError(f"Velocity checkpoint directory not found: {directory}")
    actor_path = resolve_checkpoint_actor(directory / "actor.pt")
    for path in (actor_path, actor_path.with_name("world_model.pt")):
      if not path.is_file():
        raise FileNotFoundError(f"Missing velocity checkpoint file: {path}")

    metadata = actor_path.with_name("runner_config.json")
    if metadata.is_file():
      try:
        saved = json.loads(metadata.read_text(encoding="utf-8"))["runner_cfg"]
        saved_limit = float(saved.get("action_limit", self.cfg.action_limit))
      except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid velocity checkpoint config: {metadata}") from exc
      if saved_limit != self.cfg.action_limit:
        raise ValueError(
          f"The saved actor uses --runner.action-limit {saved_limit:g}, "
          f"but the current runner uses {self.cfg.action_limit:g}"
        )
    return actor_path

  def _bind_task(self) -> None:
    term = self.env.unwrapped.command_manager.get_term("twist")
    if not isinstance(term, VelocityCommand):
      raise TypeError("The velocity runner requires a VelocityCommand term named twist")
    self.velocity_command = term
    self._sim_reward = cast(
      SimVelocityReward, self.env.unwrapped.reward_manager.get_term_cfg("velocity").func
    )

  def _task_buffer_schema(self) -> dict[str, FieldSpec]:
    return {
      "velocity_command": FieldSpec(3),
      "command_time_left": FieldSpec(),
      "velocity_reward_signals": FieldSpec(VELOCITY_REWARD_SIGNAL_COUNT),
    }

  def _transition_task_fields(self) -> dict[str, torch.Tensor]:
    return {
      "velocity_command": self.velocity_command.command.clone(),
      "command_time_left": self.velocity_command.time_left.clone(),
    }

  def _transition_task_post_fields(self) -> dict[str, torch.Tensor]:
    return {"velocity_reward_signals": self._sim_reward.signals}

  def _init_policy(self) -> None:
    representation = cast(GenCoordRepresentation, self._representation)
    sensor_dim = representation.proprio_obs_dim(self._num_actions)
    actor_dim = self.cfg.history_length * sensor_dim + self._num_goals
    critic_dim = (
      representation.rep_state_dim + self._num_goals + self._sim_context.output_dim
    )
    actor = VelocityActor(actor_dim, self._num_actions, self.cfg).to(self.device)
    if self._init_actor_path is not None:
      weights = torch.load(self._init_actor_path, map_location="cpu", weights_only=True)
      try:
        actor.load_state_dict(weights)
      except (TypeError, RuntimeError) as exc:
        raise ValueError(
          f"Actor from {self._init_actor_path} does not match the current runner"
        ) from exc
      print(
        f"[INFO] Initialized the velocity actor and world model from {self._init_actor_path.parent}"
      )
    # Start the critic fresh so it learns values for the current reward.
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
      sim_context=self._sim_context,
      reward_signals=self._reward_signals,
      amp_discriminator=self._amp.learner.discriminator
      if self._amp is not None
      else None,
      amp_cfg=self.cfg.amp if self._amp is not None else None,
      amp_features=self._amp.features if self._amp is not None else None,
    )
    if torch.device(self.device).type == "cuda":
      self.wm_env.reset = torch.compile(self.wm_env.reset)
      self.wm_env.step = torch.compile(self.wm_env.step)
    actor_lr = None
    action_regularizer = None
    if self._amp is not None and self._init_actor_path is not None:
      actor_lr = min(self.cfg.policy.lr, self.cfg.amp.warm_start_actor_lr)
      if self.cfg.amp.action_anchor_weight > 0.0:
        reference_actor = deepcopy(actor).eval().requires_grad_(False)
        reward_dt = representation.dt if self.env.cfg.scale_rewards_by_dt else 1.0
        anchor_scale = self.cfg.amp.action_anchor_weight * reward_dt

        def action_anchor(obs: TensorDict, actions: torch.Tensor) -> torch.Tensor:
          with torch.no_grad():
            reference_actions = reference_actor(obs)
          return anchor_scale * (actions - reference_actions).square().sum(dim=-1)

        action_regularizer = action_anchor
      print(f"[INFO] AMP warm-start actor learning rate: {actor_lr:g}")
    self.policy_learner = ActorCriticPolicyLearner(
      actor,
      critic,
      self.wm_env,
      self.cfg.policy,
      device=self.device,
      actor_lr=actor_lr,
      action_regularizer=action_regularizer,
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
      if (
        self._init_actor_path is not None
        or self._num_wm_updates >= self.policy_warmup_updates
      ):
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
      bundle = self._wm.bundle()
      bundle["velocity_reward_model"] = self._reward_signals.state_dict()
      if self._amp is not None:
        bundle["amp_discriminator"] = self._amp.learner.discriminator.state_dict()
        bundle["amp_discriminator_version"] = AMP_DISCRIMINATOR_VERSION
      torch.save(bundle, staging_dir / "world_model.pt")
      torch.save(self.policy_learner.actor.state_dict(), staging_dir / "actor.pt")
      torch.save(self.policy_learner.critic.state_dict(), staging_dir / "critic.pt")
    return [
      str(checkpoint_dir / name)
      for name in ("world_model.pt", "actor.pt", "critic.pt", "runner_config.json")
    ]

  def _publish_checkpoint(self, files: list[str]) -> None:
    if wandb.run is not None and wandb.run.settings.mode == "online":
      super()._publish_checkpoint(files)
