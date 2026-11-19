"""Play a SuperMimic multi-encoder actor against the sim."""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mjlab.tasks  # noqa: F401
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends

import supermimic.tasks  # noqa: F401
from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
from supermimic.runners.supermimic_runner import (
  SuperMimicRunnerCfg,
  build_actor,
  build_world_model,
)
from supermimic.runners.velocity_runner import VELOCITY_TASK_ID
from supermimic.scripts.play_velocity import VelocityPlayConfig, play_velocity
from supermimic.scripts.viewers import (
  NativeViewerStartupTimeout,
  SuperMimicNativeViewer,
  SuperMimicViserViewer,
  create_wm_vis_model,
)
from supermimic.utils.gamepad import GamepadError
from supermimic.utils.local_checkpoints import resolve_checkpoint_actor
from supermimic.utils.wandb_artifacts import (
  download_artifact_cached,
  download_model_dir,
)


@dataclass
class PlayConfig:
  wandb_path: str | None = None
  """W&B run path. Choose either this or --checkpoint."""
  checkpoint: Path | None = None
  """Local actor .pt file, with world_model.pt in the same directory."""
  runner_config: Path | None = None
  """Local runner config JSON. Defaults to the checkpoint's sibling runner_config.json."""
  motion_file: str | None = None
  """Reference motion .npz or glob. Required when using --checkpoint."""
  initial_task: Literal["motion", "velocity"] = "motion"
  """Task selected when the viewer opens. Switch live via the GUI dropdown."""
  device: str | None = None
  num_steps_rollout: int = 1000000
  viewer: Literal["auto", "native", "viser"] = "viser"


def play(task_id: str, cfg: PlayConfig) -> None:
  if (cfg.wandb_path is None) == (cfg.checkpoint is None):
    raise ValueError("Pass exactly one of --checkpoint and --wandb-path.")
  if cfg.runner_config is not None and cfg.checkpoint is None:
    raise ValueError("--runner-config requires --checkpoint.")

  actor_path = cfg.checkpoint.expanduser() if cfg.checkpoint is not None else None
  run = None
  if actor_path is not None:
    if cfg.motion_file is None:
      raise ValueError("--checkpoint requires a local --motion-file.")
    actor_path = resolve_checkpoint_actor(actor_path)
    for path in (actor_path, actor_path.with_name("world_model.pt")):
      if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    config_path = cfg.runner_config or actor_path.with_name("runner_config.json")
    config_path = config_path.expanduser()
    if cfg.runner_config is not None or config_path.is_file():
      runner_cfg = load_runner_cfg(json.loads(config_path.read_text(encoding="utf-8")))
    else:
      representation = "maxcoord" if "MaxCoord" in task_id else "gencoord"
      runner_cfg = SuperMimicRunnerCfg(representation=representation)
      print(f"[INFO] No runner_config.json found; using {representation} defaults")
    print(f"[INFO] Loading local checkpoint: {actor_path}")
  else:
    import wandb

    print(f"[INFO] Connecting to WandB run: {cfg.wandb_path}")
    run = wandb.Api().run(cfg.wandb_path)
    runner_cfg = load_runner_cfg(run.config)

  check_representation_matches(task_id, runner_cfg.representation)
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  print(f"[INFO] Loading environment: {task_id}")
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.observations["actor"].enable_corruption = True

  command_term = env_cfg.commands["motion"]
  assert isinstance(command_term, MotionCommandCfg)
  motion_names: list[str] = []
  if cfg.motion_file is not None:
    expanded = os.path.expanduser(cfg.motion_file)
    if any(c in expanded for c in "*?[]"):
      files = sorted(glob.glob(expanded))
    else:
      files = [expanded]
    if not files:
      raise ValueError(f"No motion files matched: {cfg.motion_file}")
    command_term.motion_file = files
    motion_names = [Path(f).stem for f in files]
    print(f"[INFO] Using {len(files)} local motion(s)")
  elif run is not None:
    motion_artifacts = [a for a in run.used_artifacts() if a.type == "motions"]
    if motion_artifacts:
      motion_files = []
      for artifact in motion_artifacts:
        print(f"[INFO] Found linked motion artifact: {artifact.name}")
        motion_files.append(str(download_artifact_cached(artifact) / "motion.npz"))
      command_term.motion_file = (
        motion_files if len(motion_files) > 1 else motion_files[0]
      )
      motion_names = [a.name.split(":")[0] for a in motion_artifacts]
      print(f"[INFO] Loaded {len(motion_files)} motion(s)")

  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))

  wm = build_world_model(runner_cfg, env, device)
  dynamics = wm.dynamics

  if actor_path is None:
    assert run is not None
    actor_path = download_model_dir(run) / "actor.pt"
  wm.load_bundle(
    torch.load(actor_path.with_name("world_model.pt"), map_location=device)
  )
  dynamics.eval()

  actor = build_actor(runner_cfg, wm, device)
  actor.load_state_dict(torch.load(actor_path, map_location=device))
  actor.eval()

  wm_vis_model = create_wm_vis_model(env.unwrapped.sim.mj_model)

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved = "native" if has_display else "viser"
  else:
    resolved = cfg.viewer
  print(f"[INFO] Starting viewer: {resolved}")

  if resolved == "native":
    viewer = SuperMimicNativeViewer(
      env=env,
      actor=actor,
      dynamics=dynamics,
      representation=wm.representation,
      wm_vis_model=wm_vis_model,
      num_steps_rollout=cfg.num_steps_rollout,
      sim_context=wm.sim_context,
      task_specs=actor.task_specs,
      initial_task=cfg.initial_task,
    )
  else:
    viewer = SuperMimicViserViewer(
      env=env,
      actor=actor,
      dynamics=dynamics,
      representation=wm.representation,
      wm_vis_model=wm_vis_model,
      num_steps_rollout=cfg.num_steps_rollout,
      motion_names=motion_names,
      sim_context=wm.sim_context,
      task_specs=actor.task_specs,
      initial_task=cfg.initial_task,
    )

  try:
    viewer.run()
  finally:
    env.close()


def main() -> None:
  all_tasks = sorted([*list_tasks(), VELOCITY_TASK_ID])
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  if chosen_task == VELOCITY_TASK_ID:
    args = tyro.cli(VelocityPlayConfig, args=remaining_args, config=mjlab.TYRO_FLAGS)
    try:
      play_velocity(args)
    except (
      GamepadError,
      NativeViewerStartupTimeout,
      FileNotFoundError,
      ValueError,
    ) as exc:
      raise SystemExit(str(exc)) from None
    except KeyboardInterrupt:
      return
    return
  args = tyro.cli(PlayConfig, args=remaining_args, config=mjlab.TYRO_FLAGS)
  try:
    play(chosen_task, args)
  except NativeViewerStartupTimeout as exc:
    raise SystemExit(str(exc)) from None


if __name__ == "__main__":
  main()
