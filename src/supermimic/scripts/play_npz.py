"""Play G1 reference motion in the native MuJoCo viewer."""

import argparse
import math
import time
from functools import partial
from glob import glob
from pathlib import Path
from queue import SimpleQueue
from zipfile import BadZipFile

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
from mjlab.viewer.native.keys import KEY_ENTER, KEY_LEFT, KEY_RIGHT, KEY_SPACE
from mujoco.viewer import launch_passive

from supermimic.scripts.native_viewer import (
  NativeViewerStartupTimeout,
  close_native_viewer,
  setup_native_viewer,
)


def resolve_motion_paths(sources: list[str]) -> list[Path]:
  paths: list[Path] = []
  for source in sources:
    path = Path(source).expanduser()
    if path.is_file():
      matches = [path]
    elif path.is_dir():
      matches = path.iterdir()
    else:
      matches = map(Path, glob(str(path)))
    files = sorted(p for p in matches if p.is_file() and p.suffix.lower() == ".npz")
    if not files:
      raise ValueError(f"No NPZ motion files found for {source!r}")
    paths.extend(p.resolve() for p in files)
  return list(dict.fromkeys(paths))


def build_model() -> mujoco.MjModel:
  spec = get_spec()
  spec.add_texture(
    name="floor_grid",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    width=128,
    height=128,
    rgb1=(0.23, 0.27, 0.31),
    rgb2=(0.32, 0.36, 0.40),
  )
  material = spec.add_material(name="floor", texuniform=True, texrepeat=(2, 2))
  material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_grid"
  spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.15), material="floor"
  )
  return spec.compile()


def load_motion(path: Path, model: mujoco.MjModel) -> tuple[np.ndarray, float]:
  """Load one clip as an array of poses in MuJoCo's qpos order."""
  try:
    with np.load(path, allow_pickle=False) as motion:
      fps = float(motion["fps"].item())
      root_pos = np.asarray(motion["root_pos_w"], dtype=np.float64)
      root_quat = np.asarray(motion["root_quat_w"], dtype=np.float64)
      joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
      joint_names = motion["joint_names"].astype(str)
  except (OSError, EOFError, ValueError, KeyError, TypeError, BadZipFile) as exc:
    raise ValueError(f"{path.name}: could not read the motion data ({exc})") from exc

  joint_addresses = {
    model.joint(i).name: int(model.jnt_qposadr[i])
    for i in range(model.njnt)
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
  }
  if (
    joint_names.shape != (len(joint_addresses),)
    or set(joint_names) != joint_addresses.keys()
  ):
    raise ValueError(f"{path.name}: joint_names must match the G1 joints")

  num_frames = joint_pos.shape[0] if joint_pos.ndim == 2 else 0
  if (
    not num_frames
    or root_pos.shape != (num_frames, 3)
    or root_quat.shape != (num_frames, 4)
    or joint_pos.shape != (num_frames, len(joint_addresses))
  ):
    raise ValueError(f"{path.name}: invalid or inconsistent pose array shapes")
  if not math.isfinite(fps) or fps <= 0:
    raise ValueError(f"{path.name}: fps must be finite and positive")
  if not all(np.isfinite(array).all() for array in (root_pos, root_quat, joint_pos)):
    raise ValueError(f"{path.name}: poses contain non-finite values")
  quat_norms = np.linalg.norm(root_quat, axis=1, keepdims=True)
  if np.any(quat_norms < 1e-8) or not np.isfinite(quat_norms).all():
    raise ValueError(f"{path.name}: invalid root quaternion")

  poses = np.broadcast_to(model.qpos0, (num_frames, model.nq)).copy()
  root = model.joint("floating_base_joint").qposadr.item()
  poses[:, root : root + 3] = root_pos
  poses[:, root + 3 : root + 7] = root_quat / quat_norms
  poses[:, [joint_addresses[name] for name in joint_names]] = joint_pos
  return poses, fps


def play(paths: list[Path], speed: float, once: bool) -> None:
  model = build_model()
  data = mujoco.MjData(model)
  index = 0
  poses, fps = load_motion(paths[index], model)
  data.qpos[:] = poses[0]

  keys: SimpleQueue[int] = SimpleQueue()
  open_window = partial(
    launch_passive,
    model,
    data,
    key_callback=keys.put,
    show_left_ui=False,
    show_right_ui=False,
  )
  viewer = setup_native_viewer(open_window, close_native_viewer)
  try:
    with viewer.lock():
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      viewer.cam.trackbodyid = model.body("pelvis").id
      viewer.cam.distance = 2.8
      viewer.cam.azimuth = 140
      viewer.cam.elevation = -18

    print("Space: pause/resume | Left/Right: previous/next clip | Enter: restart")
    print(f"[1/{len(paths)}] {paths[index].name} ({fps:g} FPS)", flush=True)
    paused = False
    frame_position = 0.0
    last_update = time.monotonic()
    displayed_frame = -1
    while viewer.is_running():
      now = time.monotonic()
      if not paused:
        frame_position += (now - last_update) * fps * speed
      last_update = now

      next_index = index
      while not keys.empty():
        key = keys.get_nowait()
        if key == KEY_SPACE:
          paused = not paused
        elif key == KEY_ENTER:
          frame_position = 0.0
        elif key == KEY_LEFT:
          next_index = (next_index - 1) % len(paths)
          frame_position = 0.0
        elif key == KEY_RIGHT:
          next_index = (next_index + 1) % len(paths)
          frame_position = 0.0

      if frame_position >= len(poses):
        if once and index == len(paths) - 1:
          break
        next_index = (index + 1) % len(paths)
        frame_position = 0.0

      if next_index != index:
        index = next_index
        poses, fps = load_motion(paths[index], model)
        print(
          f"[{index + 1}/{len(paths)}] {paths[index].name} ({fps:g} FPS)", flush=True
        )
        displayed_frame = -1
        last_update = time.monotonic()

      frame = int(frame_position)
      if displayed_frame != frame:
        with viewer.lock():
          data.qpos[:] = poses[frame]
        displayed_frame = frame
      viewer.sync(state_only=True)
      time.sleep(0.01)
  finally:
    close_native_viewer(viewer)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("motion", nargs="+", help="G1 NPZ file, directory, or glob")
  parser.add_argument(
    "--speed", type=float, default=1.0, help="Playback speed multiplier"
  )
  parser.add_argument("--once", action="store_true", help="Exit after the last clip")
  args = parser.parse_args()
  if not math.isfinite(args.speed) or args.speed <= 0:
    parser.error("--speed must be finite and positive")
  try:
    play(resolve_motion_paths(args.motion), args.speed, args.once)
  except (ValueError, NativeViewerStartupTimeout) as exc:
    parser.error(str(exc))
  except KeyboardInterrupt:
    return


if __name__ == "__main__":
  main()
