# `multi_grid_simple`

Isolated controller workspace for rebuilding the base `create3` model into a cleaner platform before adding:
- multiscale place cells
- grid cells
- better experiment tooling
- deeper Webots automation

Current status:
- base single-scale model is working
- automated Webots launch is working
- smoke tests and base-mode validation are working
- plotting and run summaries are working

## Environment

Run controller-side scripts from the repo-local `.venv`, not the Windows Store
`python`.

Recommended from the repo root:

```powershell
.\.venv\Scripts\python ...
```

Top-level setup instructions live in:
- [README.md](../../../README.md)

## Folder Layout

- `multi_grid_simple.py`
  - controller entrypoint used by Webots
- `config.py`
  - default mode/config values
- `driver.py`
  - runtime controller loop
- `runner.py`
  - programmatic execution layer
- `launcher.py`
  - launch Webots in automated mode
- `smoke_test.py`
  - smoke and validation harnesses
- `plot_run.py`
  - post-run plotting entrypoint
- `run_summary.py`
  - writes `summary.json` and `review.md` into each run
- `layers/`
  - local model layers
- `robot/`
  - local robot/world helpers
- `worlds/`
  - controller-local world templates
- `runs/`
  - saved run artifacts
- `visualizations/`
  - generated plots
- `scratchpad/`
  - local planning/design notes

## Modes

Supported controller modes:
- `LEARN_OJAS`
- `LEARN_HEBB`
- `DMTP`
- `EXPLOIT`
- `PLOTTING`

Default mode selection lives in [config.py](./config.py).

## How To Run

### 1. Run From Webots Normally

Point a world at controller `multi_grid_simple`, then start Webots.

The controller entrypoint is:
- [multi_grid_simple.py](./multi_grid_simple.py)

### 2. Run Through The Automation Harness

The automation harness launches Webots in batch/fast mode and injects configuration through environment variables.

Main launch/config environment variables:
- `MULTI_GRID_SIMPLE_SELECTED_MODE`
- `MULTI_GRID_SIMPLE_MODE_PARAMS_JSON`
- `MULTI_GRID_SIMPLE_EXECUTION_CONFIG_JSON`
- `MULTI_GRID_SIMPLE_AUTOMATION_CONFIG_JSON`
- `MULTI_GRID_SIMPLE_LAUNCH_CONFIG_JSON`
- `MULTI_GRID_SIMPLE_SMOKE_TEST_CONFIG_JSON`

Default unattended behavior:
- automated runs now quit Webots on completion
- they do not pause on completion unless you explicitly override that in
  `MULTI_GRID_SIMPLE_AUTOMATION_CONFIG_JSON`

Webots executable resolution order:
1. `LAUNCH_CONFIG["webots_executable"]`
2. `WEBOTS_EXECUTABLE`
3. `WEBOTS_HOME`
4. `PATH`

## Core Run Parameters

The current mode parameter surface includes:
- `run_time_hours`
- `randomize_start_loc`
- `start_loc`
- `start_rotation`
- `goal_location`
- `max_dist`
- `show_bvc_activation`
- `enable_ojas`
- `enable_stdp`
- `run_id`
- `load_networks_from_run_id`
- `load_hmaps_from_run_id`
- `runtime_profile`
- `runtime_profile_overrides`

## Runtime Profiles

Named runtime profiles live in [runtime_profiles.py](./runtime_profiles.py).

Current profiles:
- `10x10_place_small`
- `20x20_place_small`
- `20x20_place_medium`
- `20x20_place_large`

Selection behavior:
- if `runtime_profile` is set, that exact profile is used
- if `runtime_profile` is omitted, the controller chooses a world-aware default
  - `10x10_*` worlds default to `10x10_place_small`
  - `20x20_*` worlds default to `20x20_place_medium`

Override behavior:
- `runtime_profile_overrides` applies a flat key/value patch on top of the selected profile
- unknown override keys raise an error
- explicit top-level `max_dist` still overrides the profile value for that run

Each run writes both:
- `runtime_profile`
- `runtime_parameters`

into `runs/<run_id>/config.json`, so the exact resolved settings are preserved.

For future automation work, profile sweeps can be implemented by iterating the
named registry in [runtime_profiles.py](./runtime_profiles.py) rather than
hardcoding constants into the driver.

Important reuse behavior:
- `load_networks_from_run_id`
  - loads `pcn.pkl` and `rcn.pkl` from a previous run
- `load_hmaps_from_run_id`
  - seeds the current run histories from a previous run
- these are separate
  - loading networks does **not** automatically append hmaps
  - append behavior is opt-in

## Common Workflows

### Fresh Learning Run

Example:
```powershell
$env:MULTI_GRID_SIMPLE_SELECTED_MODE = "LEARN_OJAS"
$env:MULTI_GRID_SIMPLE_MODE_PARAMS_JSON = '{"run_time_hours": 1, "run_id": "learn_1hr"}'
.\.venv\Scripts\python webots/controllers/multi_grid_simple/multi_grid_simple.py
```

### Evaluate A Learned Network In A Fresh Plotting Run

This reuses learned networks but starts with empty hmaps.

```powershell
$env:MULTI_GRID_SIMPLE_SELECTED_MODE = "PLOTTING"
$env:MULTI_GRID_SIMPLE_MODE_PARAMS_JSON = '{"run_time_hours": 1, "run_id": "plot_from_learned", "load_networks_from_run_id": "learn_1hr"}'
.\.venv\Scripts\python webots/controllers/multi_grid_simple/multi_grid_simple.py
```

### Continue Accumulating Hmaps From A Prior Run

This reuses learned networks and appends prior hmaps into the new run history.

```powershell
$env:MULTI_GRID_SIMPLE_SELECTED_MODE = "PLOTTING"
$env:MULTI_GRID_SIMPLE_MODE_PARAMS_JSON = '{"run_time_hours": 1, "run_id": "plot_append", "load_networks_from_run_id": "learn_1hr", "load_hmaps_from_run_id": "learn_1hr"}'
.\.venv\Scripts\python webots/controllers/multi_grid_simple/multi_grid_simple.py
```

## Programmatic Execution

Use [runner.py](./runner.py) for in-controller execution helpers:
- single trial
- repeated trials in one Webots session
- reset/reuse flow

Use [launcher.py](./launcher.py) for out-of-process Webots launch:
- resolve source worlds from `multi_grid_simple/worlds/` or `webots/worlds/`
- prepare a temporary world copy under `webots/worlds/` pointing at `multi_grid_simple`
- launch Webots in fast batch mode
- clean up generated automation worlds after test runs

## Testing And Validation

### Smoke Suite

Runs:
- `PLOTTING`
- `LEARN_OJAS`

```powershell
.\.venv\Scripts\python -c "from webots.controllers.multi_grid_simple.smoke_test import run_smoke_test_suite; print(run_smoke_test_suite())"
```

### Single-Session Smoke Suite

Runs repeated trials in one Webots session for:
- `PLOTTING`
- `LEARN_OJAS`

```powershell
.\.venv\Scripts\python -c "from webots.controllers.multi_grid_simple.smoke_test import run_single_session_smoke_suite; print(run_single_session_smoke_suite())"
```

### Base-Mode Validation Suite

Validates:
- `LEARN_HEBB`
- `DMTP`
- `EXPLOIT`

```powershell
.\.venv\Scripts\python -c "from webots.controllers.multi_grid_simple.smoke_test import run_base_mode_validation_suite; print(run_base_mode_validation_suite())"
```

## Plotting

Generate post-run plots by `run_id`:

```powershell
.\.venv\Scripts\python webots/controllers/multi_grid_simple/plot_run.py <run_id>
```

Current outputs:
- `trajectory.png`
- `place_cells.png`
- `grid_cells.png`
- `aliasing_heatmap.png`
- `aliasing_stats.json`

Options:
- `--cells N`
- `--gridsize G`
- `--seed S`
- `--aliasing-gridsize G`
- `--aliasing-quantile Q`
- `--aliasing-min-visits N`
- `--aliasing-distance-fraction F`
- `--aliasing-distance-threshold D`
- `--aliasing-min-distance-bins N`
- `--aliasing-display-percentile P`

Example:
```powershell
.\.venv\Scripts\python webots/controllers/multi_grid_simple/plot_run.py smoke_learn_ojas --cells 0 --gridsize 80 --seed 42
```

Example profile override for a 20x20 run:
```powershell
$env:MULTI_GRID_SIMPLE_SELECTED_MODE = "LEARN_OJAS"
$env:MULTI_GRID_SIMPLE_MODE_PARAMS_JSON = '{"run_time_hours": 1, "run_id": "maze_medium", "runtime_profile": "20x20_place_medium", "start_loc": [-8, -8], "goal_location": [7, 7]}'
.\.venv\Scripts\python webots/controllers/multi_grid_simple/multi_grid_simple.py
```

Aliasing plot notes:
- the aliasing plot computes a spatial aliasing index from binned place-cell population vectors
- by default `--aliasing-gridsize 0` enables automatic grid sizing from the amount of run data
- the default far-bin threshold is chosen automatically from arena span and bin size
- `aliasing_stats.json` records the resolved grid size, MSAI, and plotting parameters

## Run Artifacts

Each run is stored under:
- `runs/<run_id>/`

Current standard files:
- `config.json`
- `metrics.json`
- `summary.json`
- `review.md`
- `hmaps/hmap_loc.pkl`
- `hmaps/hmap_pcn.pkl`
- `hmaps/hmap_hdn.pkl`
- `hmaps/hmap_bvc.pkl`
- `hmaps/hmap_gcn.pkl`
- `networks/pcn.pkl`
- `networks/rcn.pkl`

Plot outputs are stored under:
- `visualizations/<run_id>/`

## Notes

- `PLOTTING` disables learning, but it still computes place-cell activations.
- A fresh untrained PCN is not silent in this implementation.
  - it starts with random BVC-to-PC input weights
  - so it can produce spatially varying activations before Ojas learning
- if you want to visualize a network learned in a previous run, use `load_networks_from_run_id`

## Current Boundaries

This controller is still the single-scale base platform.

Not implemented yet here:
- multiscale place-only model
- multiscale plotting
- richer experiment orchestration on top of the current runner

For ongoing project-local planning, see:
- `scratchpad/`
