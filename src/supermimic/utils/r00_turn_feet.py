"""R00 foot clearance and support signals for pure-turn rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_mul

from supermimic.utils.velocity_angmom import (
  BodyKinematics,
  VelocityAngularMomentum,
  select_columns,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
type TurnFootSignals = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class R00TurnFeet:
  """Measure each foot's clearance, ground-normal angular rate, support and load."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    kinematics: VelocityAngularMomentum | None = None,
  ) -> None:
    self._robot = env.scene["robot"]
    self._sensor = env.scene["feet_ground_contact"]
    self._origins = env.scene.env_origins
    self._sim_foot_rows = tuple(self._robot.body_names.index(name) for name in _FEET)
    self._contact_rows = tuple(self._sensor.primary_names.index(name) for name in _FEET)
    self._kin_foot_rows = (
      tuple(kinematics.body_names.index(name) for name in _FEET)
      if kinematics is not None
      else None
    )

    model = env.sim.mj_model
    robot_body_ids = self._robot.indexing.body_ids.cpu().tolist()
    self._body_weight = float(
      model.body_subtreemass[robot_body_ids[0]] * np.linalg.norm(model.opt.gravity)
    )
    soles = []
    for row in self._sim_foot_rows:
      active = (model.geom_bodyid == robot_body_ids[row]) & (
        (model.geom_contype != 0) | (model.geom_conaffinity != 0)
      )
      geoms = np.flatnonzero(active)
      if len(geoms) == 0 or not np.all(
        model.geom_type[geoms] == mujoco.mjtGeom.mjGEOM_CYLINDER
      ):
        raise ValueError("R00 turn clearance requires cylindrical soles")
      soles.append(geoms)
    if len(soles[0]) != len(soles[1]):
      raise ValueError("R00 turn clearance requires the same sole layout on both feet")
    ids = np.stack(soles)
    axes = np.empty((*ids.shape, 3))
    for row, foot_geoms in enumerate(ids):
      for col, geom in enumerate(foot_geoms):
        mujoco.mju_rotVecQuat(
          axes[row, col], np.array([0.0, 0.0, 1.0]), model.geom_quat[geom]
        )
    self._centers = torch.tensor(
      model.geom_pos[ids], dtype=torch.float32, device=env.device
    )
    self._axes = torch.tensor(axes, dtype=torch.float32, device=env.device)
    self._radius = torch.tensor(
      model.geom_size[ids, 0], dtype=torch.float32, device=env.device
    )
    self._half_length = torch.tensor(
      model.geom_size[ids, 1], dtype=torch.float32, device=env.device
    )
    self._up = torch.tensor([0.0, 0.0, 1.0], device=env.device)

  def clearance_and_yaw(
    self, foot_pos: torch.Tensor, foot_quat: torch.Tensor, foot_omega: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the lowest sole surface and signed yaw rate in flat-ground axes.

    Foot positions are relative to the floor origin. Quaternions and angular
    velocities use world axes, with Z perpendicular to the floor.
    """
    up = quat_apply_inverse(foot_quat, self._up.expand_as(foot_pos)).unsqueeze(-2)
    up_along_axis = (up * self._axes).sum(dim=-1, keepdim=True)
    up_perpendicular = torch.linalg.vector_norm(up - up_along_axis * self._axes, dim=-1)
    center_height = foot_pos[..., 2:3] + (up * self._centers).sum(dim=-1)
    extent = (
      self._half_length * up_along_axis.squeeze(-1).abs()
      + self._radius * up_perpendicular
    )
    return (center_height - extent).amin(dim=-1), foot_omega[..., 2]

  def from_sim(self) -> TurnFootSignals:
    """Return L/R clearance, yaw rate, contact indicator and normal load fraction."""
    data = self._robot.data
    foot_pos, foot_quat, foot_omega = (
      select_columns(x, self._sim_foot_rows)
      for x in (data.body_link_pos_w, data.body_link_quat_w, data.body_link_ang_vel_w)
    )
    clearance, yaw = self.clearance_and_yaw(
      foot_pos - self._origins.unsqueeze(-2), foot_quat, foot_omega
    )
    contact = self._sensor.data
    assert contact.found is not None and contact.force is not None
    support = (contact.found[:, self._contact_rows] > 0).to(clearance.dtype)
    # The net force is in world axes and oriented primary -> secondary.
    load = (contact.force[:, self._contact_rows, 2].abs() / self._body_weight).clamp(
      max=1.0
    ) * support
    return clearance, yaw, support, load

  def from_kinematics(
    self, kin: BodyKinematics, root_pos: torch.Tensor, root_quat: torch.Tensor
  ) -> TurnFootSignals:
    """Return the same signals with differentiable geometric support proxies."""
    assert self._kin_foot_rows is not None
    foot_pos, foot_quat, foot_omega = (
      select_columns(x, self._kin_foot_rows) for x in (kin[0], kin[1], kin[3])
    )
    orientation = root_quat.unsqueeze(-2).expand_as(foot_quat)
    clearance, yaw = self.clearance_and_yaw(
      root_pos.unsqueeze(-2) + quat_apply(orientation, foot_pos),
      quat_mul(orientation, foot_quat),
      quat_apply(orientation, foot_omega),
    )
    # Imagined rollouts have no contact wrench. Start with full support a few
    # millimetres from the floor and let the proxy fall smoothly as a foot lifts.
    support = torch.sigmoid((0.006 - clearance) / 0.004)
    load = support / support.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return clearance, yaw, support, load
