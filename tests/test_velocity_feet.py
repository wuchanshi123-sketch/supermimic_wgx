from types import SimpleNamespace

import torch

from supermimic.utils.velocity_feet import (
  VelocityFootCosts,
  foot_ground_contact_sensor_cfg,
)


def test_velocity_foot_costs_use_matched_contact_columns_and_first_contact():
  cfg = foot_ground_contact_sensor_cfg()
  assert cfg.reduce == "netforce" and cfg.track_air_time and cfg.num_slots == 1
  assert (cfg.secondary.mode, cfg.secondary.pattern) == ("body", "terrain")

  def find_sites(names, *, preserve_order):
    assert names == ["right_foot", "left_foot"] and preserve_order
    return [1, 0], names

  def first_contact(*, dt):
    assert dt == 0.02
    return torch.tensor([[True, False], [False, True], [False, False]])

  robot = SimpleNamespace(
    find_sites=find_sites,
    data=SimpleNamespace(
      site_lin_vel_w=torch.tensor([[[3.0, 4.0, 20.0], [6.0, 8.0, 20.0]]] * 3)
    ),
  )
  sensor = SimpleNamespace(
    primary_names=["robot/right_ankle_roll_link", "robot/left_ankle_roll_link"],
    compute_first_contact=first_contact,
    data=SimpleNamespace(
      found=torch.tensor([[2, 0], [1, 1], [0, 0]]),
      force=torch.tensor([[[3.0, 4.0, 12.0], [0.0, 0.0, 38.0]]] * 3),
    ),
  )
  env = SimpleNamespace(
    scene={"robot": robot, cfg.name: sensor}, device="cpu", step_dt=0.02
  )
  slip, landing = VelocityFootCosts(env)()
  torch.testing.assert_close(slip, torch.tensor([100.0, 125.0, 0.0]))
  torch.testing.assert_close(landing, torch.tensor([13.0, 38.0, 0.0]))
