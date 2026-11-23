"""Differentiable anchor-relative body features for the AMP discriminator."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_conjugate, quat_mul

from supermimic.utils.gencoord_helpers import decompose_rep_state
from supermimic.utils.helpers import quat_to_rot6d
from supermimic.utils.velocity_angmom import (
  BodyKinematics,
  VelocityAngularMomentum,
  select_columns,
)

AMP_BODY_NAMES = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)
AMP_ANCHOR_NAME = "torso_link"
R00_AMP_BODY_NAMES = (
  "base_link",
  "neck_pitch_link",
  "left_hip_roll_link",
  "left_knee_pitch_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_pitch_link",
  "right_ankle_roll_link",
  "left_shoulder_roll_link",
  "left_elbow_pitch_link",
  "left_wrist_roll_link",
  "right_shoulder_roll_link",
  "right_elbow_pitch_link",
  "right_wrist_roll_link",
)
R00_AMP_ANCHOR_NAME = "waist_pitch_link"


class AmpBodyFeatures:
  """Use anchor-relative link poses and absolute twists in each link's axes."""

  def __init__(
    self,
    kinematics: VelocityAngularMomentum,
    *,
    body_names: tuple[str, ...] = AMP_BODY_NAMES,
    anchor_name: str = AMP_ANCHOR_NAME,
  ) -> None:
    if not body_names or len(set(body_names)) != len(body_names):
      raise ValueError("AMP body names must be nonempty and unique")
    self.kinematics = kinematics
    self.feature_dim = 15 * len(body_names)
    names = kinematics.body_names
    missing = set((*body_names, anchor_name)).difference(names)
    if missing:
      raise ValueError(f"Robot is missing AMP bodies: {sorted(missing)}")
    self._bodies = tuple(names.index(name) for name in body_names)
    self._anchor = names.index(anchor_name)

  def __call__(
    self, rep_state: torch.Tensor, kin: BodyKinematics | None = None
  ) -> torch.Tensor:
    if kin is None:
      kin = self.kinematics.body_kinematics(rep_state)
    pos, quat, vel, omega = kin
    body_pos, body_quat, body_vel, body_omega = (
      select_columns(x, self._bodies) for x in (pos, quat, vel, omega)
    )
    anchor_pos = pos[..., self._anchor : self._anchor + 1, :]
    anchor_quat = quat[..., self._anchor : self._anchor + 1, :].expand_as(body_quat)
    relative_pos = quat_apply_inverse(anchor_quat, body_pos - anchor_pos)
    relative_rot = quat_mul(quat_conjugate(anchor_quat), body_quat)

    # The shared kinematics ignore uniform translation for angular momentum.
    # AMP retains it because a forward-moving gait differs from walking in place.
    _, _, root_vel, _, _, _ = decompose_rep_state(rep_state)
    local_vel = quat_apply_inverse(body_quat, body_vel + root_vel.unsqueeze(-2))
    local_omega = quat_apply_inverse(body_quat, body_omega)
    return torch.cat(
      (
        relative_pos.flatten(-2),
        quat_to_rot6d(relative_rot).flatten(-2),
        local_vel.flatten(-2),
        local_omega.flatten(-2),
      ),
      dim=-1,
    )
