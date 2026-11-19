"""Unitree G1 Flat foot contact signals for velocity rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


_FEET_GROUND_NAME = "feet_ground_contact"
_FOOT_SITE_BY_ANKLE = {
  "left_ankle_roll_link": "left_foot",
  "right_ankle_roll_link": "right_foot",
}


def foot_ground_contact_sensor_cfg() -> ContactSensorCfg:
  """Measure one net ground-contact force per G1 foot."""
  return ContactSensorCfg(
    name=_FEET_GROUND_NAME,
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )


class VelocityFootCosts:
  """Measure ungated sliding and landing costs before the environment resets."""

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    self._robot: Entity = env.scene["robot"]
    self._sensor: ContactSensor = env.scene[_FEET_GROUND_NAME]
    # Pair each contact column with its foot site even if the entity order differs.
    sites = [
      _FOOT_SITE_BY_ANKLE[name.rsplit("/", 1)[-1]]
      for name in self._sensor.primary_names
    ]
    site_ids, _ = self._robot.find_sites(sites, preserve_order=True)
    self._site_ids = torch.tensor(site_ids, device=env.device, dtype=torch.long)
    self._step_dt = env.step_dt

  def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
    contact = self._sensor.data
    assert contact.found is not None and contact.force is not None
    foot_vel_xy = self._robot.data.site_lin_vel_w[:, self._site_ids, :2]
    slip = (foot_vel_xy.square().sum(-1) * (contact.found > 0)).sum(-1)
    first_contact = self._sensor.compute_first_contact(dt=self._step_dt)
    soft_landing = (
      torch.linalg.vector_norm(contact.force, dim=-1) * first_contact
    ).sum(-1)
    return slip, soft_landing
