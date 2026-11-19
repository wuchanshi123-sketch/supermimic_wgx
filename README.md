# SuperMimic

[Project page](https://supermimic.github.io) | [Paper](https://arxiv.org/abs/2610.28480)

SuperMimic trains a humanoid whole-body motion tracker with one differentiable world model, built on [mjlab](https://github.com/mujocolab/mjlab).

## Setup

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone both repos as siblings:

```bash
mkdir supermimic_ws && cd supermimic_ws
git clone git@github.com:HybridRobotics/supermimic.git
git clone git@github.com:HybridRobotics/mjlab_research.git
cd supermimic
```

## Usage

Available task IDs:

- `SuperMimic-GenCoord-G1` — generalized coordinates representation
- `SuperMimic-MaxCoord-G1` — maximal coordinates representation

Train a world model and policy:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> uv run src/supermimic/scripts/train.py \
  <TASK_ID> --runner.run_name <description>
```

Training saves the initial weights in `checkpoint_0/` before the first
iteration. By default, it then saves a separate checkpoint every 2,000 completed
iterations (`checkpoint_2000/`, `checkpoint_4000/`, and so on). Each checkpoint
is under `logs/<TASK_ID>/<run-timestamp>/checkpoint_<iteration>/`.
The run timestamp uses `YYYY-MM-DD_HH-MM-SS`.
The run directory links `actor.pt`, `world_model.pt`, `critic.pt` (when saved),
and `runner_config.json` to the latest checkpoint.

Play back a trained model:

```bash
uv run src/supermimic/scripts/play.py \
  <TASK_ID> --wandb-path <RUN_PATH>
```

To play a local SuperMimic checkpoint in the native MuJoCo window without W&B,
run this from a terminal in your desktop session:

```bash
uv run src/supermimic/scripts/play.py <TASK_ID> \
  --checkpoint "logs/<TASK_ID>/<run-timestamp>/actor.pt" \
  --motion-file path/to/motion.npz \
  --device cpu --viewer native
```

To play a particular iteration, point `--checkpoint` to its `actor.pt`, for
example `checkpoint_2000/actor.pt`. Keep the matching `world_model.pt` next to
`actor.pt`. Playback reads `runner_config.json` from the same directory when
available. For an older checkpoint, playback uses the task's default runner
settings unless you pass `--runner-config` with the saved `{"runner_cfg": {...}}`
JSON.

Press **Space** to pause or resume, **Enter** to reset, and **Right Arrow** to
advance one step while paused. Omit `--device cpu` to use CUDA when available.
Use `--viewer viser` to play in a browser.

### G1 velocity following

`SuperMimic-GenCoord-G1-Velocity` learns to follow sampled forward, lateral, and
yaw-rate commands on flat ground. It uses the differentiable world model and
its actor-critic learner. Training starts from the G1 standing pose and
collects its own MuJoCo data.

```bash
CUDA_VISIBLE_DEVICES=0 uv run src/supermimic/scripts/train.py \
  SuperMimic-GenCoord-G1-Velocity --num-envs 2048 \
  --runner.run-name supermimic_g1_velocity --runner.max-iterations 100000 \
  --env.sim.mujoco.iterations 50 --env.sim.mujoco.ls-iterations 50
```

The default limits are ±1.0 m/s forward and backward, ±1.0 m/s sideways, and
±3.0 rad/s in yaw. Set the ranges with
`--env.commands.twist.lin-vel-x`, `--env.commands.twist.lin-vel-y`, and
`--env.commands.twist.ang-vel-z`; pass each range as a quoted tuple, such as
`'(-1.0,1.0)'`. `Episode_Metrics/vx_error`, `vy_error`, and `wz_error` report
measured tracking errors in MuJoCo.

Drive a local velocity actor with a Linux gamepad in the native MuJoCo window.

```bash
uv run src/supermimic/scripts/play.py SuperMimic-GenCoord-G1-Velocity \
  --checkpoint "logs/SuperMimic-GenCoord-G1-Velocity/<run-timestamp>/actor.pt" \
  --gamepad-device /dev/input/js1
```

The left stick controls forward/backward and lateral velocity (both ±1.0 m/s).
The right stick controls yaw rate (±3.0 rad/s). Centered sticks and a
disconnected gamepad give a zero command. **Space** pauses or resumes, and
**Enter** resets the robot. Use `ls -l /dev/input/by-id/*-joystick` to locate a
stable device path. The player reads `actor.pt` directly and uses the sibling
`runner_config.json` to recover a custom action limit when available.

## Acknowledgements

SuperMimic contains code inspired by these projects:

- [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE) (BSD-3-Clause)
- [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) (Apache-2.0)
- [DreamerV3](https://github.com/danijar/dreamerv3) (MIT)

## License

Copyright 2026 The SuperMimic Authors.

Released under the [Apache License 2.0](LICENSE).
