"""Sagittal reflection of R00 velocity actor observations and actions."""

from collections.abc import Sequence

import torch


class R00Symmetry:
  def __init__(
    self, joint_names: Sequence[str], history_length: int, device: str | torch.device
  ) -> None:
    names = [name.rsplit("/", 1)[-1] for name in joint_names]
    if history_length < 1 or not names or len(set(names)) != len(names):
      raise ValueError(
        "R00 symmetry requires a positive history and unique joint names"
      )

    columns = {name: i for i, name in enumerate(names)}
    partners, signs = [], []
    for name in names:
      side, _, suffix = name.partition("_")
      if side in ("left", "right"):
        opposite = "right" if side == "left" else "left"
        partner = f"{opposite}_{suffix}"
      else:
        partner = name  # Waist and neck are on the symmetry plane.
      if partner not in columns or not name.endswith(
        ("_pitch_joint", "_roll_joint", "_yaw_joint")
      ):
        raise ValueError(f"Unsupported or unpaired R00 joint: {name}")
      partners.append(columns[partner])
      # Hinge rotation axes are axial vectors, so pitch keeps its sign.
      signs.append(1 if name.endswith("_pitch_joint") else -1)

    self._action_indices = torch.tensor(partners, dtype=torch.long, device=device)
    self._action_signs = torch.tensor(signs, dtype=torch.int8, device=device)

    # Polar gravity and axial gyro, followed by q_rel, dq, previous raw action.
    n = len(names)
    frame_indices = list(range(6))
    for offset in (6, 6 + n, 6 + 2 * n):
      frame_indices.extend(offset + partner for partner in partners)
    frame_signs = [1, -1, 1, -1, 1, -1] + signs * 3
    frame_width = len(frame_indices)
    obs_indices = [
      frame * frame_width + column
      for frame in range(history_length)
      for column in frame_indices
    ]
    # The current [vx, vy, wz] appears once, after all history frames.
    obs_indices.extend(
      range(history_length * frame_width, history_length * frame_width + 3)
    )
    self._obs_indices = torch.tensor(obs_indices, dtype=torch.long, device=device)
    self._obs_signs = torch.tensor(
      frame_signs * history_length + [1, -1, -1], dtype=torch.int8, device=device
    )

  def mirror_obs(self, obs: torch.Tensor) -> torch.Tensor:
    """Reflect the raw sensor and command vector before actor normalization."""
    if obs.shape[-1] != self._obs_indices.numel():
      raise ValueError("Actor observation width does not match the R00 symmetry layout")
    return obs.index_select(-1, self._obs_indices) * self._obs_signs

  def mirror_action(self, action: torch.Tensor) -> torch.Tensor:
    """Reflect actions in policy joint order, before applying scale and offset."""
    if action.shape[-1] != self._action_indices.numel():
      raise ValueError("Actor action width does not match the R00 joint count")
    return action.index_select(-1, self._action_indices) * self._action_signs
