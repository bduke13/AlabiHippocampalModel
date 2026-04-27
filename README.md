# GridMultiscaleNew

Webots-based navigation experiments built around place cells, boundary vector cells, reward cells, and grid-cell analysis. The repo currently contains:

- `webots/controllers/multi_grid_simple`
  - the cleaner single-scale controller workspace
- `webots/controllers/multiscale_grid_controller`
  - the older feature-rich multiscale research controller
- `visualizations/`
  - legacy analysis and plotting scripts

## Recommended Python Setup

Use a repo-local `uv` environment pinned to Python `3.12`.

Why:

- the Windows Store `python` on this machine is not the same environment used by all project tooling
- some scripts need `torch`
- Webots automation and local plotting are much easier to keep consistent with one explicit interpreter

### 1. Install `uv`

Preferred on Windows:

```powershell
winget install --id=astral-sh.uv -e
```

Alternative:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart the shell after install, then verify:

```powershell
uv --version
```

### 2. Create the project environment

From the repo root:

```powershell
uv python install 3.12
uv venv --python 3.12
uv pip sync --index-strategy unsafe-best-match requirements-cu126.txt
```

This repo includes:

- [.python-version](./.python-version)
- [pyproject.toml](./pyproject.toml)
- [requirements.txt](./requirements.txt)
- [requirements-cu126.txt](./requirements-cu126.txt)

`requirements-cu126.txt` wraps the pinned requirements with the PyTorch CUDA 12.6 wheel index.
The `unsafe-best-match` flag is required here because the environment intentionally mixes PyPI with the PyTorch CUDA wheel index.

### 3. Verify the environment

```powershell
.\.venv\Scripts\python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
```

The interpreter path should point at `.venv`, not the Windows Store Python path.

## Webots Prerequisites

You need a working Webots installation.

The automation harness resolves the executable in this order:

1. `LAUNCH_CONFIG["webots_executable"]`
2. `WEBOTS_EXECUTABLE`
3. `WEBOTS_HOME`
4. `PATH`

If needed, set one of:

```powershell
$env:WEBOTS_EXECUTABLE = "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"
```

or:

```powershell
$env:WEBOTS_HOME = "C:\Program Files\Webots"
```

## Daily Usage

Prefer the repo-local interpreter explicitly:

```powershell
.\.venv\Scripts\python ...
```

That avoids falling back to the Windows Store interpreter.

## `multi_grid_simple`

The current single-scale workflow is documented in:

- [webots/controllers/multi_grid_simple/README.md](./webots/controllers/multi_grid_simple/README.md)

Typical entrypoints:

- controller entrypoint:
  - [multi_grid_simple.py](./webots/controllers/multi_grid_simple/multi_grid_simple.py)
- automation runner:
  - [launcher.py](./webots/controllers/multi_grid_simple/launcher.py)
  - [runner.py](./webots/controllers/multi_grid_simple/runner.py)
- post-run plotting:
  - [plot_run.py](./webots/controllers/multi_grid_simple/plot_run.py)

Example plotting command:

```powershell
.\.venv\Scripts\python webots/controllers/multi_grid_simple/plot_run.py <run_id>
```

Example smoke test command:

```powershell
.\.venv\Scripts\python -c "from webots.controllers.multi_grid_simple.smoke_test import run_smoke_test_suite; print(run_smoke_test_suite())"
```

## Troubleshooting

### `torch` missing when a script is run

You are almost certainly using the wrong interpreter.

Check:

```powershell
python -c "import sys; print(sys.executable)"
.\.venv\Scripts\python -c "import sys; print(sys.executable)"
```

Use the `.venv` one for project scripts.

### Webots run behaves differently from local plotting

Make sure both are using the same project dependencies. The safest path is:

- run local scripts with `.\.venv\Scripts\python`
- keep Webots automation launches initiated from that same environment when possible

### Automation run fails when Webots is already open

That can cause port reuse issues or leave a prior simulation instance in the way. Close the other run before starting a new automated batch test.
