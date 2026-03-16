import json
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from config import get_smoke_test_config
    from launcher import launch_webots, prepare_controller_world
except ImportError:  # Support package import from repo root.
    from .config import get_smoke_test_config
    from .launcher import launch_webots, prepare_controller_world

CONTROLLER_DIR = Path(__file__).resolve().parent
RUNS_DIR = CONTROLLER_DIR / "runs"


@dataclass(frozen=True)
class SmokeTestResult:
    mode: str
    run_id: str
    run_dir: Path


def _mode_overrides(mode: str, smoke_config: dict, run_id: str) -> dict:
    return {
        "run_time_hours": smoke_config["run_time_hours"],
        "randomize_start_loc": False,
        "start_loc": smoke_config["start_loc"],
        "goal_location": smoke_config["goal_location"],
        "max_dist": smoke_config["max_dist"],
        "show_bvc_activation": False,
        "run_id": run_id,
    }


def _automation_overrides() -> dict:
    return {
        "quit_on_completion": True,
        "pause_on_completion": False,
        "export_image_on_completion": False,
        "completion_image_path": None,
    }


def _verify_run_artifacts(mode: str, run_dir: Path) -> None:
    required = [
        run_dir / "config.json",
        run_dir / "metrics.json",
        run_dir / "hmaps" / "hmap_loc.pkl",
        run_dir / "hmaps" / "hmap_pcn.pkl",
        run_dir / "hmaps" / "hmap_hdn.pkl",
        run_dir / "hmaps" / "hmap_bvc.pkl",
    ]
    if mode == "LEARN_OJAS":
        required.extend(
            [
                run_dir / "networks" / "pcn.pkl",
                run_dir / "networks" / "rcn.pkl",
            ]
        )

    missing = [path for path in required if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise RuntimeError(f"Smoke test for {mode} missing expected artifacts: {missing_str}")


def run_smoke_test_suite() -> list[SmokeTestResult]:
    smoke_config = get_smoke_test_config()
    world_copy = prepare_controller_world(
        smoke_config["world"],
        controller_name="multi_grid_simple",
        suffix="smoke",
    )

    results: list[SmokeTestResult] = []

    for mode in smoke_config["modes"]:
        run_id = f"smoke_{mode.lower()}"
        run_dir = RUNS_DIR / run_id
        if run_dir.exists():
            # Keep old artifacts from masking failures.
            for child in run_dir.rglob("*"):
                if child.is_file():
                    child.unlink()

        env = {
            "MULTI_GRID_SIMPLE_SELECTED_MODE": mode,
            "MULTI_GRID_SIMPLE_MODE_PARAMS_JSON": json.dumps(
                _mode_overrides(mode, smoke_config, run_id)
            ),
            "MULTI_GRID_SIMPLE_AUTOMATION_CONFIG_JSON": json.dumps(
                _automation_overrides()
            ),
        }

        process = launch_webots(world_copy, env=env)
        try:
            process.wait(timeout=smoke_config["timeout_seconds"])
        except Exception:
            process.kill()
            process.wait()
            raise RuntimeError(f"Smoke test for {mode} timed out.")

        if process.returncode not in (0, None):
            raise RuntimeError(f"Smoke test for {mode} failed with exit code {process.returncode}.")

        _verify_run_artifacts(mode, run_dir)
        results.append(SmokeTestResult(mode=mode, run_id=run_id, run_dir=run_dir))

        time.sleep(1.0)

    return results


if __name__ == "__main__":
    results = run_smoke_test_suite()
    for result in results:
        print(f"{result.mode}: {result.run_dir}")
