"""Differentiable MuJoCo subtree angular momentum for velocity training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_mul

from supermimic.utils.gencoord_helpers import decompose_rep_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(frozen=True)
class _BodyLevel:
  parents: tuple[int, ...]
  joints: tuple[int, ...]
  constants: torch.Tensor


def _select_columns(x: torch.Tensor, columns: tuple[int, ...]) -> torch.Tensor:
  """Pack static columns without differentiating through an index gather."""
  return torch.stack([x[..., i, :] for i in columns], dim=-2)


class VelocityAngularMomentum:
  """Squared angular momentum of the G1 root subtree about its center of mass."""

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    robot = env.scene["robot"]
    model = env.sim.mj_model
    device = env.device
    body_ids = robot.indexing.body_ids.cpu().tolist()
    root, *remaining = body_ids
    cols = {
      address: column
      for column, address in enumerate(robot.indexing.joint_q_adr.cpu().tolist())
    }
    bodies = [root]
    parents = [root]
    levels = []
    while remaining:
      current = [b for b in remaining if int(model.body_parentid[b]) in parents]
      if not current:
        raise ValueError("Expected all G1 links to descend from its pelvis")

      joints = []
      constants = []
      for body in current:
        num_joints = model.body_jntnum[body]
        fixed_q = model.body_quat[body]
        anchor, axis, turn_q = np.zeros(3), np.zeros(3), np.zeros(4)
        if num_joints:
          joint = model.body_jntadr[body]
          if num_joints != 1 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError("Expected fixed or single-hinge links in the G1 subtree")
          address = int(model.jnt_qposadr[joint])
          joints.append(cols[address])
          qpos0 = model.qpos0[address]
          anchor = model.jnt_pos[joint]
          local_axis = model.jnt_axis[joint]
          mujoco.mju_rotVecQuat(axis, local_axis, fixed_q)
          mujoco.mju_mulQuat(turn_q, fixed_q, np.r_[0.0, local_axis])
        else:
          joints.append(len(cols))  # The zero column for fixed links.
          qpos0 = 0.0
        offset = np.empty(3)
        mujoco.mju_rotVecQuat(offset, anchor, fixed_q)
        constants.append(
          np.r_[model.body_pos[body] + offset, anchor, axis, fixed_q, turn_q, qpos0]
        )

      levels.append(
        _BodyLevel(
          tuple(parents.index(int(model.body_parentid[b])) for b in current),
          tuple(joints),
          torch.tensor(np.stack(constants), dtype=torch.float32, device=device),
        )
      )
      bodies.extend(current)
      parents = current
      remaining = [b for b in remaining if b not in current]
    self._levels = tuple(levels)

    com_event = env.cfg.events.get("base_com")
    com_body = (
      com_event.params["asset_cfg"].body_names[0]
      if com_event is not None
      else "torso_link"
    )
    self.com_body_id = body_ids[robot.body_names.index(com_body)]
    com_row = bodies.index(self.com_body_id)
    self._ipos = torch.tensor(
      model.body_ipos[bodies], dtype=torch.float32, device=device
    )
    self._default_com = self._ipos[com_row]
    self._com_mask = (torch.arange(len(bodies), device=device) == com_row).unsqueeze(-1)
    self._iquat = torch.tensor(
      model.body_iquat[bodies], dtype=torch.float32, device=device
    )
    self._inertia = torch.tensor(
      model.body_inertia[bodies], dtype=torch.float32, device=device
    )
    self._mass = torch.tensor(
      model.body_mass[bodies], dtype=torch.float32, device=device
    ).unsqueeze(-1)
    self._total_mass = self._mass.sum()

  def __call__(
    self, rep_state: torch.Tensor, com_ipos: torch.Tensor | None = None
  ) -> torch.Tensor:
    """Return squared angular momentum using local root and joint velocities.

    ``com_ipos`` overrides the body-local inertial position of the link selected
    by the CoM randomization event. Root rotation and uniform translation leave
    angular-momentum magnitude about the subtree CoM invariant.
    """
    _, _, _, root_omega, joint_pos, joint_vel = decompose_rep_state(rep_state)
    joints = torch.cat((joint_pos.unsqueeze(-1), joint_vel.unsqueeze(-1)), dim=-1)
    joints = torch.nn.functional.pad(joints, (0, 0, 0, 1))
    pos = torch.zeros_like(root_omega).unsqueeze(-2)
    quat = rep_state.new_tensor([1.0, 0.0, 0.0, 0.0]).expand(*pos.shape[:-1], 4)
    vel = pos
    omega = root_omega.unsqueeze(-2)
    positions, quats, velocities, angular_velocities = [pos], [quat], [vel], [omega]
    for level in self._levels:
      parent_pos, parent_quat, parent_vel, parent_omega = (
        _select_columns(x, level.parents) for x in (pos, quat, vel, omega)
      )
      q, dq = _select_columns(joints, level.joints).split(1, dim=-1)
      offset, anchor, axis, fixed_q, turn_q, qpos0 = level.constants.split(
        (3, 3, 3, 4, 4, 1), dim=-1
      )
      half = (q - qpos0) * 0.5
      quat = quat_mul(parent_quat, half.cos() * fixed_q + half.sin() * turn_q)
      anchor_from_parent = quat_apply(parent_quat, offset.expand_as(parent_pos))
      anchor_from_child = quat_apply(quat, anchor.expand_as(parent_pos))
      omega = parent_omega + quat_apply(parent_quat, axis.expand_as(parent_omega)) * dq
      pos = parent_pos + anchor_from_parent - anchor_from_child
      vel = (
        parent_vel
        + parent_omega.cross(anchor_from_parent, dim=-1)
        - omega.cross(anchor_from_child, dim=-1)
      )
      positions.append(pos)
      quats.append(quat)
      velocities.append(vel)
      angular_velocities.append(omega)

    pos, quat, vel, omega = (
      torch.cat(xs, dim=-2) for xs in (positions, quats, velocities, angular_velocities)
    )
    body_ipos = self._ipos
    if com_ipos is not None:
      body_ipos = body_ipos + self._com_mask * (com_ipos - self._default_com).unsqueeze(
        -2
      )
    com_from_body = quat_apply(quat, body_ipos.expand_as(pos))
    com_pos = pos + com_from_body
    com_vel = vel + omega.cross(com_from_body, dim=-1)
    subtree_com = (self._mass * com_pos).sum(dim=-2, keepdim=True) / self._total_mass

    inertia_quat = quat_mul(quat, self._iquat.expand_as(quat))
    spin = quat_apply(
      inertia_quat, self._inertia * quat_apply_inverse(inertia_quat, omega)
    )
    orbital = (com_pos - subtree_com).cross(self._mass * com_vel, dim=-1)
    return (spin + orbital).sum(dim=-2).square().sum(dim=-1)
