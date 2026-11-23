"""Reference-state resets for command-driven walking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers import EventTermCfg, ManagerTermBase
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

from supermimic.utils.amp_reference import AmpReference
from supermimic.utils.gencoord_helpers import decompose_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


REFERENCE_STATE_RESET_TERM = "reference_state_reset"


def reference_state_reset_event(motion_file: str, probability: float) -> EventTermCfg:
  """Build a reset event that runs after the nominal reset events."""
  return EventTermCfg(
    func=ReferenceStateReset,
    mode="reset",
    params={"motion_file": motion_file, "probability": probability},
  )


class ReferenceStateReset(ManagerTermBase):
  """Replace a random subset of nominal resets with perturbed reference poses."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    probability = cfg.params["probability"]
    if not 0.0 <= probability <= 1.0:
      raise ValueError("Reference-state reset probability must be between 0 and 1")

    self._robot = env.scene["robot"]
    self.reference = AmpReference.load(
      cfg.params["motion_file"], self._robot.joint_names, env.step_dt, env.device
    )
    # Vibe R00 reset perturbations in world axes: XYZ, roll/pitch/yaw, a
    # linear velocity increment, and an angular velocity increment.
    self._root_noise = self.reference.states.new_tensor(
      (0.02, 0.02, 0.01, 0.05, 0.05, 0.10, 0.10, 0.10, 0.05, 0.10, 0.10, 0.10)
    )
    print(
      f"[INFO] Reference-state reset probability: {probability:g} "
      f"({self.reference.num_clips} clips)"
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    *,
    motion_file: str,
    probability: float,
  ) -> None:
    del motion_file  # The reference library is loaded once when binding the term.
    if probability == 0.0 or len(env_ids) == 0:
      return
    if probability < 1.0:
      env_ids = env_ids[torch.rand(len(env_ids), device=self.device) < probability]
    count = len(env_ids)
    if count == 0:
      return

    # Unlike a discriminator transition, a reset may use each clip's last frame.
    clips = torch.randint(self.reference.num_clips, (count,), device=self.device)
    lengths = self.reference.clip_lengths[clips]
    offsets = (torch.rand(count, device=self.device) * lengths).long()
    frames = self.reference.clip_starts[clips] + offsets.minimum(lengths - 1)
    states = self.reference.states[frames]
    reference_pos, reference_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel = (
      decompose_state(states)
    )

    root_noise = self._root_noise * states.new_empty((count, 12)).uniform_(-1.0, 1.0)
    root_pos = env.scene.env_origins[env_ids] + root_noise[:, :3]
    # Keep the sampled height and center horizontal positions in their scenes.
    root_pos[:, 2] += reference_pos[:, 2]
    delta_orientation = quat_from_euler_xyz(*root_noise[:, 3:6].unbind(-1))
    root_orientation = quat_mul(delta_orientation, reference_quat)
    root_state = torch.cat(
      (
        root_pos,
        root_orientation,
        root_lin_vel + root_noise[:, 6:9],
        root_ang_vel + root_noise[:, 9:12],
      ),
      -1,
    )

    joint_pos = joint_pos + torch.empty_like(joint_pos).uniform_(-0.02, 0.02)
    lower, upper = self._robot.data.soft_joint_pos_limits[env_ids].unbind(-1)
    self._robot.write_joint_state_to_sim(
      joint_pos.clamp(lower, upper), joint_vel, env_ids=env_ids
    )
    self._robot.write_root_state_to_sim(root_state, env_ids=env_ids)
