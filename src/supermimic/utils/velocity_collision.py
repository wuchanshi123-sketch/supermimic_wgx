"""Physical self-collision measurement and learned velocity reward signals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import self_collision_cost
from torch import nn

from supermimic.models import build_mlp

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


SELF_COLLISION_HISTORY = 4
VELOCITY_REWARD_SIGNAL_COUNT = 4
_SELF_COLLISION_NAME = "self_collision"
_SELF_COLLISION_THRESHOLD = 10.0
_ACCELERATION_SQUARED_SCALE = 1.0e5
_FOOT_SLIP_SCALE = 0.1
_LANDING_FORCE_SCALE = 100.0


def self_collision_sensor_cfg(root_body_name: str = "pelvis") -> ContactSensorCfg:
  """Detect contacts within the robot's root subtree."""
  return ContactSensorCfg(
    name=_SELF_COLLISION_NAME,
    primary=ContactMatch(mode="subtree", pattern=root_body_name, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=root_body_name, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=SELF_COLLISION_HISTORY,
  )


def self_collision_count(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Count substeps during which a retained self-contact force exceeded 10 N."""
  return self_collision_cost(
    env, sensor_name=_SELF_COLLISION_NAME, force_threshold=_SELF_COLLISION_THRESHOLD
  )


class VelocityRewardSignalModel(nn.Module):
  """Predict joint acceleration, self-collision count, foot slip and landing force.

  MuJoCo provides the targets at reward time. The GenCoord state omits final
  physics-substep acceleration and contact forces, so the world model learns
  the signals from the current state, the applied action and simulation context.
  """

  def __init__(
    self,
    rep_state_dim: int,
    num_actions: int,
    sim_ctx_dim: int,
    collision_history_length: int = SELF_COLLISION_HISTORY,
  ) -> None:
    super().__init__()
    if collision_history_length < 1:
      raise ValueError("Collision history must contain at least one physics step")
    self._collision_history_length = collision_history_length
    self.network = build_mlp(
      rep_state_dim + num_actions + sim_ctx_dim,
      (128, 64),
      output_dim=VELOCITY_REWARD_SIGNAL_COUNT,
    )
    # Keep the initial estimate nearly neutral, independent of the action.
    # First collect real targets and fit the model before optimizing policy.
    with torch.no_grad():
      output = self.network[-1]
      output.weight.zero_()
      output.bias.copy_(torch.tensor([-2.0, -5.0, -4.0, -5.0]))

  def load_checkpoint(
    self, state_dict: dict[str, torch.Tensor], *, add_damping: bool = False
  ) -> None:
    """Preserve existing reward predictions when adding inputs or outputs."""
    if add_damping:
      state_dict = {
        **state_dict,
        "network.0.weight": F.pad(state_dict["network.0.weight"], (0, 1)),
      }
    layer = f"network.{len(self.network) - 1}"
    weight_key, bias_key = f"{layer}.weight", f"{layer}.bias"
    saved_weight = state_dict.get(weight_key)
    saved_bias = state_dict.get(bias_key)
    output = self.network[-1]
    if (
      saved_weight is not None
      and saved_bias is not None
      and saved_weight.shape == (2, output.in_features)
      and saved_bias.shape == (2,)
    ):
      state_dict = {
        **state_dict,
        weight_key: torch.cat(
          [saved_weight, output.weight[2:].detach().to(saved_weight)]
        ),
        bias_key: torch.cat([saved_bias, output.bias[2:].detach().to(saved_bias)]),
      }
    self.load_state_dict(state_dict)

  def forward(
    self, rep_state: torch.Tensor, raw_action: torch.Tensor, sim_ctx: torch.Tensor
  ) -> torch.Tensor:
    scores = self.network(torch.cat((rep_state, raw_action, sim_ctx), dim=-1))
    joint_acc_l2 = F.softplus(scores[..., 0]) * _ACCELERATION_SQUARED_SCALE
    collision_count = scores[..., 1].sigmoid() * self._collision_history_length
    foot_slip = F.softplus(scores[..., 2]) * _FOOT_SLIP_SCALE
    soft_landing = F.softplus(scores[..., 3]) * _LANDING_FORCE_SCALE
    return torch.stack((joint_acc_l2, collision_count, foot_slip, soft_landing), dim=-1)

  def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Fit nonnegative continuous signals and a bounded expected collision count."""
    pred_acc, pred_collision, pred_slip, pred_landing = predictions.unbind(-1)
    target_acc, target_collision, target_slip, target_landing = targets.unbind(-1)
    collision_loss = F.binary_cross_entropy(
      pred_collision / self._collision_history_length,
      target_collision / self._collision_history_length,
    )
    continuous_loss = (
      _scaled_signal_loss(pred_acc, target_acc, _ACCELERATION_SQUARED_SCALE)
      + _scaled_signal_loss(pred_slip, target_slip, _FOOT_SLIP_SCALE)
      + _scaled_signal_loss(pred_landing, target_landing, _LANDING_FORCE_SCALE)
    )
    return continuous_loss + collision_loss


def _scaled_signal_loss(
  predicted: torch.Tensor, target: torch.Tensor, scale: float
) -> torch.Tensor:
  # Log scaling limits the influence of isolated contact spikes.
  return F.smooth_l1_loss(torch.log1p(predicted / scale), torch.log1p(target / scale))
