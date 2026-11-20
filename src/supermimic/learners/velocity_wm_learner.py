"""Fit reward signals that the velocity task's GenCoord state omits."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from tensordict import TensorDict

from supermimic.learners.amp_style import AmpStyleLearner
from supermimic.learners.learner_utils import (
  CapturedUpdate,
  _optimize_step,
  build_optimizers,
  graph_state,
)
from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.models import Dynamics
from supermimic.representations import StateRepresentation
from supermimic.utils.amp_body_features import AmpBodyFeatures
from supermimic.utils.amp_reference import AmpReference
from supermimic.utils.velocity_collision import VelocityRewardSignalModel


@dataclass(frozen=True)
class VelocityAmpTraining:
  learner: AmpStyleLearner
  reference: AmpReference
  features: AmpBodyFeatures
  expert_features: torch.Tensor


class VelocityWorldModelLearner(WorldModelLearner):
  def __init__(
    self,
    dynamics: Dynamics,
    representation: StateRepresentation,
    cfg: WorldModelLearnerCfg,
    device: str,
    reward_signals: VelocityRewardSignalModel,
    *,
    amp: VelocityAmpTraining | None = None,
  ) -> None:
    super().__init__(dynamics, representation, cfg, device)
    self._reward_signals = reward_signals
    self._amp = amp
    self._signal_params = list(reward_signals.parameters())
    self._signal_optimizers = build_optimizers([reward_signals], cfg.lr, device=device)
    self._fit_signals = CapturedUpdate(
      self._fit_reward_signals,
      lambda: graph_state([reward_signals], self._signal_optimizers),
    )
    reward_signals.requires_grad_(False)

  def update(self, data: Iterable[TensorDict]) -> dict[str, torch.Tensor]:
    windows = list(data)
    logs = super().update(windows)
    # These targets describe just one transition, so use each window's start.
    batch = torch.stack(
      [
        window[:, 0].select("state", "action", "sim_ctx", "velocity_reward_signals")
        for window in windows
      ]
    )
    self._reward_signals.requires_grad_(True)
    try:
      logs = logs | self._fit_signals(batch)
    finally:
      # Policy BPTT differentiates through the frozen predictor's inputs.
      self._reward_signals.requires_grad_(False)
    if self._amp is not None:
      logs |= self._update_amp(windows, self._amp)
    return logs

  def _update_amp(
    self,
    windows: list[TensorDict],
    amp: VelocityAmpTraining,
  ) -> dict[str, torch.Tensor]:
    # The existing replay windows guarantee same-episode adjacent pairs.
    # The motion prior observes real transitions from the full command mix.
    learner = amp.learner
    pairs = torch.cat([window["state"][:, :2] for window in windows])
    indices = torch.randint(
      len(pairs),
      (learner.cfg.num_batches, learner.cfg.batch_size),
      device=pairs.device,
    )
    totals: dict[str, torch.Tensor] = {}
    for batch_ids in indices.unbind(0):
      with torch.no_grad():
        policy_features = amp.features(self.representation.represent(pairs[batch_ids]))
        starts = amp.reference.sample_indices(learner.cfg.batch_size)
        expert_features = torch.stack(
          (amp.expert_features[starts], amp.expert_features[starts + 1]), dim=1
        )
      for key, value in learner.update(expert_features, policy_features).items():
        totals[key] = totals[key] + value if key in totals else value
    return {key: value / learner.cfg.num_batches for key, value in totals.items()}

  def _fit_reward_signals(self, batch: TensorDict) -> dict[str, torch.Tensor]:
    rep_state = self._compiled_represent(batch["state"])
    predicted = self._reward_signals(rep_state, batch["action"], batch["sim_ctx"])
    loss = self._reward_signals.loss(predicted, batch["velocity_reward_signals"])
    _optimize_step(self._signal_optimizers, loss, self._signal_params, max_norm=1.0)
    return {"Loss/velocity_reward_signals": loss.detach()}
