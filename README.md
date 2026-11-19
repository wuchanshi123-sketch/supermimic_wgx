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

To train explicit G1 velocity following, pass these options to `train.py` with
a reference motion dataset.

```bash
--runner.slot-mix.motion 0.5 --runner.slot-mix.velocity 0.5 \
  --runner.tasks.velocity.mode follow
```

The velocity slots choose between reference commands and independently
sampled commands at reset and every 5 seconds by default. Independent commands
sample across bins of the configured `[vx, vy, wz]` ranges, including zero
commands. The velocity slots use their own reward and critic. Only reference
commands train latent imitation, and the velocity objective updates the velocity
encoder through the shared decoder's detached parameters.

Use `Velocity/vx_mae`, `Velocity/vy_mae` and `Velocity/wz_mae` to see the
measured MuJoCo speed errors. `Velocity/reference_bc_fraction` reports the
fraction of velocity samples eligible for latent imitation. In Viser, select
**velocity** and leave **Replay
(motion-derived)** off to test fixed commands and changes between commands.
The default `imitation` mode uses the original motion-derived velocity training.

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

On Linux, you can drive a trained velocity policy with a gamepad in the native
MuJoCo window.

```bash
uv run src/supermimic/scripts/play.py SuperMimic-GenCoord-G1 \
  --checkpoint path/to/actor.pt --motion-file path/to/standing.npz \
  --device cpu --viewer native --gamepad-device /dev/input/js1
```

Push the left stick to walk and strafe, and move the right stick horizontally to
turn. Centered sticks and a disconnected device send a zero velocity command.
The lateral command spans -1.0 to 1.0 m/s. Forward and turning commands scale to
the corresponding ranges in the checkpoint's runner config. The reference clip
supplies the reset pose, and walking can continue
after that clip's last frame. Use `ls -l /dev/input/by-id/*-joystick` to locate
the gamepad. For a nonstandard layout, `--gamepad-axes '(0, 1, 3)'` sets the raw
axis indices for left X, left Y and right X.

## Acknowledgements

SuperMimic contains code inspired by these projects:

- [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE) (BSD-3-Clause)
- [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) (Apache-2.0)
- [DreamerV3](https://github.com/danijar/dreamerv3) (MIT)

## License

Copyright 2026 The SuperMimic Authors.

Released under the [Apache License 2.0](LICENSE).
