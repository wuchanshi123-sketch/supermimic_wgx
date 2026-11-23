"""R00 31-DoF hardware for the shared flat-ground velocity task."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from supermimic.tasks.velocity_env_cfg import (
  SimVelocityReward,
  VelocityCommandCfg,
  configure_velocity_env,
)
from supermimic.utils.unitree_velocity_reward import (
  UnitreeVelocityPosture,
  UnitreeVelocityRewardCfg,
)
from supermimic.utils.velocity_reward import VelocityTerminationCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnvCfg


R00_ROOT_BODY = "base_link"
R00_TORSO_BODY = "waist_pitch_link"
_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "r00"
_SOLE_GEOMS = tuple(
  f"{side}_leg6_ankle_r_link_collision_sole_sphere_{number}"
  for side in ("l", "r")
  for number in range(1, 7)
)

# Vibe's R00 serial-model position control, grouped by joint name.
# Values are (Kp, Kd, torque limit, reflected armature).
_PD = {
  r".*_hip_(pitch|roll|yaw)_joint": (93.659, 9.937, 240.0, 0.0659),
  r".*_knee_pitch_joint": (144.823, 15.366, 320.0, 0.1019),
  r".*_ankle_pitch_joint": (40.0, 1.0, 45.4, 0.0048),
  r".*_ankle_roll_joint": (40.0, 1.0, 32.4, 0.0048),
  "waist_yaw_joint": (93.659, 9.937, 240.0, 0.0659),
  "waist_roll_joint": (40.0, 2.0, 27.0, 0.0048),
  "waist_pitch_joint": (40.0, 2.0, 53.8, 0.0048),
  r".*_(shoulder|elbow)_pitch_joint": (9.474820, 0.603186, 50.0, 0.0024),
  r".*_(shoulder_(roll|yaw)|elbow_yaw)_joint": (7.500899, 0.477522, 42.0, 0.0019),
  r".*_wrist_(pitch|roll)_joint": (21.792087, 1.387327, 12.0, 0.00552),
  r"neck_(yaw|pitch)_joint": (30.0, 1.0, 5.3, 0.01),
}
_ACTION_SCALES = {
  r".*_hip_(pitch|roll|yaw)_joint": 0.50,
  r".*_knee_pitch_joint": 0.44,
  r".*_ankle_pitch_joint": 0.23,
  r".*_ankle_roll_joint": 0.16,
  "waist_yaw_joint": 0.50,
  "waist_roll_joint": 0.135,
  "waist_pitch_joint": 0.27,
  r".*_shoulder_pitch_joint": 1.0,
  r".*_shoulder_roll_joint": 0.187,
  r".*_shoulder_yaw_joint": 1.0,
  r".*_elbow_pitch_joint": 0.675,
  r".*_elbow_yaw_joint": 1.0,
  r".*_wrist_(pitch|roll)_joint": 0.092,
  r"neck_(yaw|pitch)_joint": 0.20,
}
_POSTURE_WIDTHS = (
  ("hip_pitch", 0.5, 0.5),
  ("hip_roll", 0.15, 0.25),
  ("hip_yaw", 0.15, 0.25),
  ("knee", 0.5, 0.5),
  ("ankle_pitch", 0.15, 0.25),
  ("ankle_roll", 0.1, 0.1),
  ("waist_yaw", 0.15, 0.25),
  ("waist_roll", 0.1, 0.1),
  ("waist_pitch", 0.1, 0.1),
  ("shoulder_pitch", 0.15, 0.25),
  ("shoulder_roll", 0.1, 0.1),
  ("shoulder_yaw", 0.1, 0.1),
  ("elbow", 0.1, 0.1),
  ("wrist", 0.1, 0.1),
  ("neck", 0.1, 0.1),
)


def get_r00_spec() -> mujoco.MjSpec:
  """Load the serial robot and attach sensors required by velocity control."""
  spec = mujoco.MjSpec.from_file(str(_ASSET_DIR / "r00_serial.xml"))
  spec.assets = {
    mesh.file: (_ASSET_DIR / mesh.file).read_bytes() for mesh in spec.meshes
  }
  # The scene's MujocoCfg applies the runtime options from the flat task.
  spec.option = mujoco.MjSpec().option
  # The scene owns the ground and lighting, while mjlab supplies position PD.
  for element in (*spec.actuators, *spec.lights, spec.geom("floor")):
    spec.delete(element)
  for joint in spec.joints:
    if joint.type != mujoco.mjtJoint.mjJNT_FREE:
      joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE
      joint.actfrcrange[:] = (0.0, 0.0)
      joint.damping[:] = 0.0
      joint.frictionloss = 0.0

  for geom in spec.geoms:
    if geom.name in _SOLE_GEOMS:
      # Let the randomized sole friction take precedence over the plane.
      geom.priority = 1
  for side in ("left", "right"):
    spec.body(f"{side}_ankle_roll_link").add_site(
      name=f"{side}_foot",
      pos=(
        0.0365,
        0.0,
        -0.068,
      ),  # Midpoint of the toe-heel support span at sole height.
      size=(0.005, 0.005, 0.005),
      rgba=(0.0, 0.0, 0.0, 0.0),
    )
  spec.body("base_imu_frame").add_site(
    name="imu_in_base", size=(0.005, 0.005, 0.005), rgba=(0.0, 0.0, 0.0, 0.0)
  )
  spec.add_sensor(
    name="imu_ang_vel",
    type=mujoco.mjtSensor.mjSENS_GYRO,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu_in_base",
  )
  spec.add_sensor(
    name="root_angmom",
    type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    objname=R00_ROOT_BODY,
  )
  return spec


def get_r00_robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_r00_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.96),
      joint_pos={
        "left_shoulder_roll_joint": 0.05,
        "right_shoulder_roll_joint": -0.05,
        r".*_elbow_pitch_joint": -0.2,
      },
      joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=tuple(
        BuiltinPositionActuatorCfg(
          target_names_expr=(pattern,),
          stiffness=kp,
          damping=kd,
          effort_limit=effort,
          armature=armature,
        )
        for pattern, (kp, kd, effort, armature) in _PD.items()
      ),
      soft_joint_pos_limit_factor=0.9,
    ),
  )


class R00SimVelocityReward(SimVelocityReward):
  torso_body_name = R00_TORSO_BODY

  def _make_posture(self, robot: Entity) -> UnitreeVelocityPosture:
    return UnitreeVelocityPosture(
      robot.joint_names, robot.data.default_joint_pos[0], joint_widths=_POSTURE_WIDTHS
    )


def r00_velocity_env_cfg(
  play: bool = False, *, amp: bool = False
) -> ManagerBasedRlEnvCfg:
  """Build the flat R00 task with ranges for the selected mode."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_r00_robot_cfg()}
  cfg.viewer.body_name = R00_TORSO_BODY
  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = dict(_ACTION_SCALES)
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _SOLE_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = (R00_TORSO_BODY,)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.025, 0.025),
    1: (-0.05, 0.05),
    2: (-0.05, 0.05),
  }
  cfg.events["reset_base"].params["pose_range"]["z"] = (-0.01, 0.01)
  command_cfg = VelocityCommandCfg(
    lin_vel_x=(-0.5, 1.0),
    lin_vel_y=(-0.5, 0.5),
    ang_vel_z=(-3.14, 3.14) if play or amp else (-1.0, 1.0),
    external=play,
  )
  return configure_velocity_env(
    cfg,
    play=play,
    reward_cfg=UnitreeVelocityRewardCfg(angular_momentum_cost=0.0)
    if amp
    else UnitreeVelocityRewardCfg(),
    fall_cfg=VelocityTerminationCfg(min_height=0.50, max_tilt_rad=math.radians(70.0)),
    command_cfg=command_cfg,
    root_body_name=R00_ROOT_BODY,
    reward_term=R00SimVelocityReward,
  )
