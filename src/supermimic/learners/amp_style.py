"""Least-squares AMP on adjacent G1 body features."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from supermimic.learners.learner_utils import _optimize_step


@dataclass(frozen=True)
class AmpStyleCfg:
  motion_file: str | None = None
  """Reference NPZ file, directory, or glob. None disables AMP."""
  rsi_probability: float = 0.0
  """Probability of initializing a real simulation episode from a reference frame."""
  weight: float = 1.25
  """Style reward rate. At 50 Hz, the maximum contribution per step is 0.025."""
  lr: float = 1e-4
  gradient_penalty: float = 10.0
  batch_size: int = 512
  num_batches: int = 2
  hidden_dims: tuple[int, ...] = (1024, 512, 256)
  style_command_gate: float = 0.2
  """Command magnitude at which the style reward reaches full strength."""
  style_speed_gate: bool = False
  """Attenuate the style reward by planar and yaw tracking errors."""
  warm_start_actor_lr: float = 1e-5
  """Actor learning-rate cap when fine-tuning a saved velocity policy with AMP."""
  action_anchor_weight: float = 0.25
  """Warm-start penalty weight on squared action deviation from the saved actor."""
  mirror_consistency: bool = False
  """Enable the R00 actor's sagittal mirror consistency loss."""
  mirror_weight: float = 1.0
  """Per-step reward-rate weight on mean squared mirror error in raw actions."""

  def __post_init__(self) -> None:
    if not 0.0 <= self.rsi_probability <= 1.0:
      raise ValueError("AMP rsi_probability must be between zero and one")
    for name in ("lr", "warm_start_actor_lr"):
      value = getattr(self, name)
      if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"AMP {name} must be finite and positive")
    for name in (
      "weight",
      "gradient_penalty",
      "style_command_gate",
      "action_anchor_weight",
      "mirror_weight",
    ):
      value = getattr(self, name)
      if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"AMP {name} must be finite and nonnegative")
    if self.batch_size < 1 or self.num_batches < 1:
      raise ValueError("AMP batch size and number of batches must be positive")
    if any(width < 1 for width in self.hidden_dims):
      raise ValueError("AMP hidden dimensions must be positive")


def amp_command_gate(command: torch.Tensor, threshold: float) -> torch.Tensor:
  """Ramp style by planar speed and half the yaw-command magnitude."""
  if threshold == 0.0:
    return torch.ones_like(command[..., 0])
  magnitude = (
    torch.linalg.vector_norm(command[..., :2], dim=-1) + 0.5 * command[..., 2].abs()
  )
  return (magnitude / threshold).clamp(0.0, 1.0)


class AmpDiscriminator(nn.Module):
  """Score a transition from its two precomputed body-feature frames."""

  def __init__(self, feature_dim: int, cfg: AmpStyleCfg) -> None:
    super().__init__()
    if feature_dim < 1:
      raise ValueError("AMP feature dimension must be positive")
    self.feature_dim = feature_dim
    width = 2 * feature_dim
    layers: list[nn.Module] = []
    for hidden in cfg.hidden_dims:
      layers.extend((nn.Linear(width, hidden), nn.ReLU()))
      width = hidden
    layers.append(nn.Linear(width, 1))
    self.network = nn.Sequential(*layers)
    self.register_buffer("feature_mean", torch.zeros(feature_dim))
    self.register_buffer("feature_var", torch.ones(feature_dim))
    self.register_buffer("feature_count", torch.zeros((), dtype=torch.float64))

  @torch.no_grad()
  def fit_normalizer(self, expert_frames: torch.Tensor) -> None:
    """Initialize statistics on the reference before collecting policy data."""
    self.feature_count.zero_()
    self.update_normalizer(expert_frames)

  @torch.no_grad()
  def update_normalizer(self, frames: torch.Tensor) -> None:
    """Merge statistics without rebinding buffers read by captured rollouts."""
    if frames.ndim != 2 or frames.shape[-1] != self.feature_dim or frames.shape[0] == 0:
      raise ValueError(
        f"Expected nonempty AMP features of shape [N, {self.feature_dim}]"
      )
    variance, mean = torch.var_mean(frames.float(), dim=0, correction=0)
    count = frames.shape[0]
    rate = (count / (self.feature_count + count)).float()
    delta = mean - self.feature_mean
    self.feature_mean.add_(rate * delta)
    self.feature_var.copy_(
      (1.0 - rate) * self.feature_var
      + rate * variance
      + rate * (1.0 - rate) * delta.square()
    )
    self.feature_count.add_(count)

  def normalized_features(self, frames: torch.Tensor) -> torch.Tensor:
    return (
      (frames.float() - self.feature_mean) / (self.feature_var + 1.0e-4).sqrt()
    ).clamp(-10.0, 10.0)

  def forward(self, current: torch.Tensor, following: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=current.device.type, enabled=False):
      pair = torch.cat(
        (self.normalized_features(current), self.normalized_features(following)), dim=-1
      )
      return self.network(pair).squeeze(-1)

  def reward(self, current: torch.Tensor, following: torch.Tensor) -> torch.Tensor:
    return self.reward_from_score(self(current, following))

  @staticmethod
  def reward_from_score(score: torch.Tensor) -> torch.Tensor:
    quadratic = (1.0 - 0.25 * (score - 1.0).square()).clamp_min(0.0)
    # Match Vibe's reward and slope at the join while preserving an actor
    # gradient near the negative label, where the hard clip would be flat.
    join = -0.5
    join_value = 1.0 - 0.25 * (join - 1.0) ** 2
    join_slope = 0.5 * (1.0 - join)
    tail = join_value * torch.exp(
      (join_slope / join_value) * (score - join).clamp_max(0.0)
    )
    return torch.where(score < join, tail, quadratic)


class AmpStyleLearner:
  """Fit the prior on adjacent real replay and expert body-feature frames."""

  def __init__(
    self, discriminator: AmpDiscriminator, cfg: AmpStyleCfg, device: str
  ) -> None:
    self.discriminator = discriminator.to(device)
    self.cfg = cfg
    self._params = list(discriminator.parameters())
    self._optimizer = torch.optim.Adam(
      [
        {"params": discriminator.network[:-1].parameters(), "weight_decay": 1e-3},
        {"params": discriminator.network[-1].parameters(), "weight_decay": 0.1},
      ],
      lr=cfg.lr,
    )
    discriminator.requires_grad_(False)

  def update(
    self, expert_features: torch.Tensor, policy_features: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    """Fit from [B, 2, feature_dim] batches, preserving input gradients at use time."""
    discriminator = self.discriminator
    feature_dim = discriminator.feature_dim
    if any(
      x.ndim != 3 or x.shape[1:] != (2, feature_dim) or x.shape[0] == 0
      for x in (expert_features, policy_features)
    ):
      raise ValueError(
        f"Expected nonempty adjacent AMP features of shape [B, 2, {feature_dim}]"
      )

    expert = expert_features.detach().float()
    policy = policy_features.detach().float()
    all_frames = torch.cat((expert.flatten(0, 1), policy.flatten(0, 1)))
    if not bool(torch.isfinite(all_frames).all()):
      raise FloatingPointError("Non-finite AMP training sample")
    discriminator.update_normalizer(all_frames)
    expert = discriminator.normalized_features(expert).flatten(-2)
    policy = discriminator.normalized_features(policy).flatten(-2)
    discriminator.requires_grad_(True)
    try:
      with (
        torch.enable_grad(),
        torch.autocast(device_type=expert.device.type, enabled=False),
      ):
        expert.requires_grad_(self.cfg.gradient_penalty > 0.0)
        expert_score = discriminator.network(expert).squeeze(-1)
        policy_score = discriminator.network(policy).squeeze(-1)
        classification = 0.5 * (
          (expert_score - 1.0).square().mean() + (policy_score + 1.0).square().mean()
        )
        penalty = classification.new_zeros(())
        if self.cfg.gradient_penalty > 0.0:
          (gradient,) = torch.autograd.grad(
            expert_score.sum(), expert, create_graph=True
          )
          penalty = self.cfg.gradient_penalty * gradient.square().sum(-1).mean()
        loss = classification + penalty
        grad_norm = _optimize_step([self._optimizer], loss, self._params, max_norm=1.0)
        assert grad_norm is not None
    finally:
      self._optimizer.zero_grad(set_to_none=True)
      # Freeze weights for policy BPTT while preserving gradients through the input.
      discriminator.requires_grad_(False)

    return {
      "AMP/loss": loss.detach(),
      "AMP/discriminator_loss": classification.detach(),
      "AMP/gradient_penalty": penalty.detach(),
      "AMP/grad_norm": grad_norm.detach(),
      "AMP/expert_score": expert_score.detach().mean(),
      "AMP/policy_score": policy_score.detach().mean(),
      "AMP/style_reward": discriminator.reward_from_score(policy_score.detach()).mean(),
    }
