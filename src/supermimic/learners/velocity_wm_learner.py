"""Fit reward signals that the velocity task's GenCoord state omits."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from tensordict import TensorDict

from supermimic.learners.learner_utils import (
  CapturedUpdate,
  _optimize_step,
  build_optimizers,
  graph_state,
)
from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.models import Dynamics
from supermimic.representations import StateRepresentation
from supermimic.utils.velocity_collision import VelocityRewardSignalModel


class VelocityWorldModelLearner(WorldModelLearner):
  def __init__(
    self,
    dynamics: Dynamics,
    representation: StateRepresentation,
    cfg: WorldModelLearnerCfg,
    device: str,
    reward_signals: VelocityRewardSignalModel,
  ) -> None:
    super().__init__(dynamics, representation, cfg, device)
    self._reward_signals = reward_signals
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
      return logs | self._fit_signals(batch)
    finally:
      # Policy BPTT differentiates through the frozen predictor's inputs.
      self._reward_signals.requires_grad_(False)

  def _fit_reward_signals(self, batch: TensorDict) -> dict[str, torch.Tensor]:
    rep_state = self._compiled_represent(batch["state"])
    predicted = self._reward_signals(rep_state, batch["action"], batch["sim_ctx"])
    loss = self._reward_signals.loss(predicted, batch["velocity_reward_signals"])
    _optimize_step(self._signal_optimizers, loss, self._signal_params, max_norm=1.0)
    return {"Loss/velocity_reward_signals": loss.detach()}
