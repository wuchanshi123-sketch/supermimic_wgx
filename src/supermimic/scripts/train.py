"""Script to train World Model using SuperMimicRunner."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import mjlab.tasks  # noqa: F401
import tyro
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.runners.velocity_runner import (
  R00_VELOCITY_AMP_TASK_ID,
  R00_VELOCITY_TASK_IDS,
  VELOCITY_AMP_TASK_IDS,
  VELOCITY_TASK_ID,
  VELOCITY_TASK_IDS,
  VelocityRunner,
  VelocityRunnerCfg,
)
from supermimic.scripts.train_utils import (
  create_env,
  download_registry_motions,
  setup_device,
)
from supermimic.tasks.r00_velocity_env_cfg import r00_velocity_env_cfg
from supermimic.tasks.velocity_env_cfg import configure_turn_reward, velocity_env_cfg
from supermimic.tasks.velocity_rsi import (
  REFERENCE_STATE_RESET_TERM,
  reference_state_reset_event,
)


@dataclass
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  runner: SuperMimicRunnerCfg

  # Experiment & Environment
  num_envs: int = 4096
  device: str | None = None
  registry_name: str | None = "supermimic/wandb-registry-motions/lafan_run1_subject5"
  motion_file: str | None = None

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id, play=False)
    return TrainConfig(
      env=env_cfg,
      runner=SuperMimicRunnerCfg(),
    )


def main() -> None:
  # Parse first argument to choose the task.
  all_tasks = sorted([*list_tasks(), *VELOCITY_TASK_IDS])
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  if chosen_task in VELOCITY_TASK_IDS:
    train_velocity(remaining_args, chosen_task)
    return
  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    config=mjlab.TYRO_FLAGS,
  )

  device = setup_device(args.device)
  print(f"[INFO] Loading environment: {chosen_task}")

  if args.motion_file:
    # Local motions handed straight to MotionLoader (globs/concatenates).
    if "," in args.motion_file:
      args.env.commands["motion"].motion_file = [
        os.path.expanduser(p.strip()) for p in args.motion_file.split(",")
      ]
    else:
      args.env.commands["motion"].motion_file = os.path.expanduser(args.motion_file)
    args.registry_name = None  # avoid linking a stale registry artifact for lineage
    print(f"[INFO] Using local motion file: {args.motion_file}")
  else:
    motion_file = download_registry_motions(args.registry_name)
    if motion_file is not None:
      args.env.commands["motion"].motion_file = motion_file

  # Auto-detect representation from task_id if not explicitly set
  if "MaxCoord" in chosen_task and args.runner.representation == "gencoord":
    args.runner.representation = "maxcoord"
    print("[INFO] Auto-detected maxcoord representation from task_id")

  args.env.commands["motion"].sampling_mode = (
    "external" if args.runner.sampling_mode == "return" else args.runner.sampling_mode
  )

  env = create_env(args.env, args.num_envs, device)

  # Init Runner
  runner = SuperMimicRunner(
    env=env,
    cfg=args.runner,
    task_id=chosen_task,
    registry_name=args.registry_name,
    device=device,
  )

  runner.run()


@dataclass
class VelocityTrainConfig:
  env: ManagerBasedRlEnvCfg = field(default_factory=velocity_env_cfg)
  runner: VelocityRunnerCfg = field(default_factory=VelocityRunnerCfg)
  num_envs: int = 2048
  device: str | None = None


def train_velocity(remaining_args: list[str], task_id: str = VELOCITY_TASK_ID) -> None:
  amp = task_id in VELOCITY_AMP_TASK_IDS
  r00 = task_id in R00_VELOCITY_TASK_IDS
  env_cfg = r00_velocity_env_cfg if r00 else velocity_env_cfg
  default = VelocityTrainConfig(env=env_cfg(amp=amp))
  if task_id == R00_VELOCITY_AMP_TASK_ID:
    default.runner.turn_reward = replace(default.runner.turn_reward, enabled=True)
  if amp:
    motions = (
      Path.home() / "workspace/supermimic_vibe/data/processed/r00_amp_50hz.npz"
      if r00
      else Path(__file__).resolve().parents[4] / "WalkandRun_amp_walk_50hz"
    )
    default.runner.amp = replace(
      default.runner.amp,
      motion_file=str(motions),
      rsi_probability=0.5 if r00 else 0.0,
      mirror_consistency=r00,
    )
  args = tyro.cli(
    VelocityTrainConfig, args=remaining_args, default=default, config=mjlab.TYRO_FLAGS
  )
  amp_cfg = args.runner.amp
  turn_cfg = args.runner.turn_reward
  if turn_cfg.enabled and task_id != R00_VELOCITY_AMP_TASK_ID:
    raise ValueError("Pure-turn rewards require the R00 velocity AMP task")
  if task_id == R00_VELOCITY_AMP_TASK_ID:
    configure_turn_reward(args.env, turn_cfg)
  if amp_cfg.rsi_probability > 0.0:
    if not amp_cfg.motion_file:
      raise ValueError("--runner.amp.rsi-probability requires --runner.amp.motion-file")
    # Run after nominal resets so only the selected environments are overwritten.
    args.env.events[REFERENCE_STATE_RESET_TERM] = reference_state_reset_event(
      amp_cfg.motion_file, amp_cfg.rsi_probability
    )
  device = setup_device(args.device)
  print(f"[INFO] Loading environment: {task_id}")
  env = create_env(args.env, args.num_envs, device)
  VelocityRunner(env, args.runner, task_id=task_id, device=device).run()


if __name__ == "__main__":
  main()
