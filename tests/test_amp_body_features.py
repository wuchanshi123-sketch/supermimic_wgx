"""Compare AMP body features with MuJoCo link kinematics."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

from supermimic.utils.amp_body_features import (
  AMP_ANCHOR_NAME,
  AMP_BODY_NAMES,
  R00_AMP_ANCHOR_NAME,
  R00_AMP_BODY_NAMES,
  AmpBodyFeatures,
)
from supermimic.utils.gencoord_helpers import compute_gencoord_state
from supermimic.utils.velocity_angmom import VelocityAngularMomentum


def _r00_layout_model() -> mujoco.MjModel:
  # Verify the R00 feature layout without depending on external robot assets.
  limbs = "".join(
    f'<body name="{name}" pos="0.1 {0.01 * index} 0.05">'
    '<joint axis="1 0 0"/><geom size="0.01"/></body>'
    for index, name in enumerate(R00_AMP_BODY_NAMES[1:])
  )
  return mujoco.MjModel.from_xml_string(
    f'<mujoco><worldbody><body name="base_link"><freejoint/><geom size="0.1"/>'
    f'<body name="{R00_AMP_ANCHOR_NAME}" pos="0 0.1 0.25">'
    f'<joint axis="0 1 0"/><geom size="0.05"/>{limbs}'
    "</body></body></worldbody></mujoco>"
  )


@pytest.mark.parametrize("robot_name", ["g1", "r00"])
def test_amp_features_match_link_poses_and_twists(robot_name):
  if robot_name == "r00":
    model = _r00_layout_model()
    body_names, anchor_name = R00_AMP_BODY_NAMES, R00_AMP_ANCHOR_NAME
    feature_dim = 210
  else:
    model = mujoco.MjModel.from_xml_path(str(G1_XML))
    body_names, anchor_name = AMP_BODY_NAMES, AMP_ANCHOR_NAME
    feature_dim = 195
  body_ids = torch.arange(1, model.nbody)
  robot = SimpleNamespace(
    indexing=SimpleNamespace(body_ids=body_ids, joint_q_adr=torch.arange(7, model.nq)),
    body_names=tuple(model.body(i).name for i in body_ids.tolist()),
  )
  env = SimpleNamespace(
    scene={"robot": robot},
    sim=SimpleNamespace(mj_model=model),
    cfg=SimpleNamespace(
      events={
        "base_com": SimpleNamespace(
          params={"asset_cfg": SimpleNamespace(body_names=[anchor_name])}
        )
      }
    ),
    device="cpu",
  )
  kinematics = VelocityAngularMomentum(env)
  extract = (
    AmpBodyFeatures(kinematics, body_names=body_names, anchor_name=anchor_name)
    if robot_name == "r00"
    else AmpBodyFeatures(kinematics)
  )
  assert extract.feature_dim == feature_dim
  link_ids = [model.body(name).id for name in body_names]
  torso_id = model.body(anchor_name).id
  data = mujoco.MjData(model)
  rng = np.random.default_rng(37)
  states, expected = [], []
  for _ in range(6):
    data.qpos[:] = rng.normal(size=model.nq)
    data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
    data.qvel[:] = rng.normal(size=model.nv)
    mujoco.mj_forward(model, data)
    world_rotations = data.xmat.reshape(-1, 3, 3)
    states.append(
      np.r_[
        data.qpos, data.qvel[:3], world_rotations[1] @ data.qvel[3:6], data.qvel[6:]
      ]
    )

    torso_t = world_rotations[torso_id].T
    rel_pos = (data.xpos[link_ids] - data.xpos[torso_id]) @ torso_t.T
    rel_rot = (torso_t @ world_rotations[link_ids])[..., :2]
    twist = np.zeros((len(link_ids), 6))
    for i, link_id in enumerate(link_ids):
      mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_XBODY, link_id, twist[i], 1
      )
    expected.append(
      np.r_[
        rel_pos.ravel(), rel_rot.ravel(), twist[:, 3:].ravel(), twist[:, :3].ravel()
      ]
    )

  state = compute_gencoord_state(torch.tensor(np.stack(states), dtype=torch.float32))
  state = state.reshape(2, 3, -1).requires_grad_()
  result = extract(state, kinematics.body_kinematics(state))
  torch.testing.assert_close(result, extract(state))
  torch.testing.assert_close(
    result,
    torch.tensor(np.stack(expected), dtype=torch.float32).reshape(2, 3, feature_dim),
    atol=5e-6,
    rtol=1e-5,
  )
  (grad,) = torch.autograd.grad(result.square().sum(), state)
  assert torch.isfinite(grad).all()
  assert torch.count_nonzero(grad[..., 4:7]) > 0  # Pelvis translation influences AMP.
  assert torch.count_nonzero(grad[..., 7:10]) > 0
  assert torch.count_nonzero(grad[..., 10:]) > 0
