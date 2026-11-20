"""Differentiable torso-relative G1 body features for the AMP discriminator."""

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


class AmpBodyFeatures:
  """Use torso-relative link poses and absolute twists in each link's axes."""

  feature_dim = 15 * len(AMP_BODY_NAMES)

  def __init__(self, kinematics: VelocityAngularMomentum) -> None:
    self.kinematics = kinematics
    names = kinematics.body_names
    self._bodies = tuple(names.index(name) for name in AMP_BODY_NAMES)
    self._torso = names.index(AMP_ANCHOR_NAME)

  def __call__(
    self, rep_state: torch.Tensor, kin: BodyKinematics | None = None
  ) -> torch.Tensor:
    if kin is None:
      kin = self.kinematics.body_kinematics(rep_state)
    pos, quat, vel, omega = kin
    body_pos, body_quat, body_vel, body_omega = (
      select_columns(x, self._bodies) for x in (pos, quat, vel, omega)
    )
    torso_pos = pos[..., self._torso : self._torso + 1, :]
    torso_quat = quat[..., self._torso : self._torso + 1, :].expand_as(body_quat)
    relative_pos = quat_apply_inverse(torso_quat, body_pos - torso_pos)
    relative_rot = quat_mul(quat_conjugate(torso_quat), body_quat)

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
