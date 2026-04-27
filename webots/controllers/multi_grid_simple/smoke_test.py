import json
import shutil
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

try:
    from config import MODE_PARAMS, get_smoke_test_config
    from launcher import (
        canonical_world_name,
        cleanup_generated_world,
        cleanup_stale_generated_worlds,
        launch_webots,
        prepare_controller_world,
    )
    from run_summary import update_run_summary
except ImportError:  # Support package import from repo root.
    from .config import MODE_PARAMS, get_smoke_test_config
    from .launcher import (
        canonical_world_name,
        cleanup_generated_world,
        cleanup_stale_generated_worlds,
        launch_webots,
        prepare_controller_world,
    )
    from .run_summary import update_run_summary

CONTROLLER_DIR = Path(__file__).resolve().parent
RUNS_DIR = CONTROLLER_DIR / "runs"
HMAP_ARTIFACTS = (
    "hmaps/hmap_loc.pkl",
    "hmaps/hmap_pcn.pkl",
    "hmaps/hmap_hdn.pkl",
    "hmaps/hmap_bvc.pkl",
    "hmaps/hmap_gcn.pkl",
)


@dataclass(frozen=True)
class SmokeTestResult:
    mode: str
    run_id: str
    run_dir: Path


@dataclass(frozen=True)
class SessionSmokeTestResult:
    mode: str
    run_ids: list[str]
    run_dirs: list[Path]


@dataclass(frozen=True)
class ModeValidationSpec:
    mode: str
    mode_params_overrides: dict
    expected_completion_reason: str
    required_artifacts: tuple[str, ...]
    forbidden_artifacts: tuple[str, ...]
    expected_saved_files: tuple[str, ...]
    expected_config_values: dict


@dataclass(frozen=True)
class ValidationResult:
    mode: str
    run_id: str
    run_dir: Path
    completion_reason: str
    expected_artifacts: list[str]
    files_saved: list[str]


def _mode_overrides(
    mode: str,
    smoke_config: dict,
    run_id: str | None = None,
    extra_overrides: dict | None = None,
) -> dict:
    overrides = dict(MODE_PARAMS[mode])
    overrides.update(
        {
            "run_time_hours": smoke_config["run_time_hours"],
            "randomize_start_loc": False,
            "start_loc": smoke_config["start_loc"],
            "goal_location": smoke_config["goal_location"],
            "max_dist": smoke_config["max_dist"],
            "show_bvc_activation": False,
        }
    )
    if extra_overrides:
        overrides.update(extra_overrides)
    if run_id is not None:
        overrides["run_id"] = run_id
    return overrides


def _automation_overrides() -> dict:
    return {
        "quit_on_completion": True,
        "pause_on_completion": False,
        "export_image_on_completion": False,
        "completion_image_path": None,
    }


def _series_execution_overrides(smoke_config: dict, run_id_prefix: str) -> dict:
    return {
        "type": "series",
        "num_trials": int(smoke_config.get("series_num_trials", 2)),
        "reload_world_between_trials": False,
        "pause_on_completion": False,
        "run_id_prefix": run_id_prefix,
    }


def _reset_run_dir(run_dir: Path) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)


def _wait_for_run_completion(process, *, timeout_seconds: int, label: str) -> None:
    try:
        process.wait(timeout=timeout_seconds)
    except Exception:
        process.kill()
        process.wait()
        raise RuntimeError(f"{label} timed out.")

    if process.returncode not in (0, None):
        raise RuntimeError(f"{label} failed with exit code {process.returncode}.")


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _normalize_run_relative_path(run_dir: Path, path_value: str | Path) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        return path.as_posix()
    return path.resolve().relative_to(run_dir.resolve()).as_posix()


def _verify_expected_paths(run_dir: Path, relative_paths: tuple[str, ...], *, label: str) -> None:
    missing = [str(run_dir / relative_path) for relative_path in relative_paths if not (run_dir / relative_path).exists()]
    if missing:
        raise RuntimeError(f"{label} missing expected artifacts: {', '.join(missing)}")


def _verify_absent_paths(run_dir: Path, relative_paths: tuple[str, ...], *, label: str) -> None:
    present = [str(run_dir / relative_path) for relative_path in relative_paths if (run_dir / relative_path).exists()]
    if present:
        raise RuntimeError(f"{label} produced unexpected artifacts: {', '.join(present)}")


def _verify_run_artifacts(mode: str, run_dir: Path) -> None:
    required = ["config.json", "metrics.json", *HMAP_ARTIFACTS]
    if mode in ("LEARN_OJAS", "LEARN_HEBB"):
        required.extend(
            [
                "networks/pcn.pkl",
                "networks/rcn.pkl",
            ]
        )

    _verify_expected_paths(run_dir, tuple(required), label=f"Smoke test for {mode}")


def _build_validation_spec(mode: str, smoke_config: dict) -> ModeValidationSpec:
    goal_location = list(smoke_config["goal_location"])
    mode_defaults = dict(MODE_PARAMS[mode])

    if mode == "LEARN_HEBB":
        return ModeValidationSpec(
            mode=mode,
            mode_params_overrides={},
            expected_completion_reason="time_limit_reached",
            required_artifacts=("config.json", "metrics.json", *HMAP_ARTIFACTS, "networks/pcn.pkl", "networks/rcn.pkl"),
            forbidden_artifacts=(),
            expected_saved_files=(*HMAP_ARTIFACTS, "networks/pcn.pkl", "networks/rcn.pkl"),
            expected_config_values={
                "enable_ojas": mode_defaults["enable_ojas"],
                "enable_stdp": mode_defaults["enable_stdp"],
                "start_loc": list(smoke_config["start_loc"]),
                "goal_location": goal_location,
            },
        )

    if mode == "DMTP":
        return ModeValidationSpec(
            mode=mode,
            mode_params_overrides={"start_loc": goal_location},
            expected_completion_reason="goal_reached_dmtp",
            required_artifacts=("config.json", "metrics.json", "networks/rcn.pkl"),
            forbidden_artifacts=("networks/pcn.pkl", *HMAP_ARTIFACTS),
            expected_saved_files=("networks/rcn.pkl",),
            expected_config_values={
                "enable_ojas": mode_defaults["enable_ojas"],
                "enable_stdp": mode_defaults["enable_stdp"],
                "start_loc": goal_location,
                "goal_location": goal_location,
            },
        )

    if mode == "EXPLOIT":
        return ModeValidationSpec(
            mode=mode,
            mode_params_overrides={"start_loc": goal_location},
            expected_completion_reason="goal_reached_exploit",
            required_artifacts=("config.json", "metrics.json", "networks/rcn.pkl"),
            forbidden_artifacts=("networks/pcn.pkl", *HMAP_ARTIFACTS),
            expected_saved_files=("networks/rcn.pkl",),
            expected_config_values={
                "enable_ojas": mode_defaults["enable_ojas"],
                "enable_stdp": mode_defaults["enable_stdp"],
                "start_loc": goal_location,
                "goal_location": goal_location,
            },
        )

    raise ValueError(f"Unsupported validation mode '{mode}'.")


def _validate_mode_run(spec: ModeValidationSpec, run_id: str, run_dir: Path) -> ValidationResult:
    _verify_expected_paths(run_dir, spec.required_artifacts, label=f"Validation for {spec.mode}")
    _verify_absent_paths(run_dir, spec.forbidden_artifacts, label=f"Validation for {spec.mode}")

    config = _load_json(run_dir / "config.json")
    metrics = _load_json(run_dir / "metrics.json")

    expected_config = {
        "run_id": run_id,
        "mode": spec.mode,
        **spec.expected_config_values,
    }
    for key, expected_value in expected_config.items():
        actual_value = config.get(key)
        if actual_value != expected_value:
            raise RuntimeError(
                f"Validation for {spec.mode} expected config[{key!r}]={expected_value!r}, got {actual_value!r}."
            )

    expected_metrics = {
        "run_id": run_id,
        "mode": spec.mode,
        "status": "completed",
        "completion_reason": spec.expected_completion_reason,
    }
    for key, expected_value in expected_metrics.items():
        actual_value = metrics.get(key)
        if actual_value != expected_value:
            raise RuntimeError(
                f"Validation for {spec.mode} expected metrics[{key!r}]={expected_value!r}, got {actual_value!r}."
            )

    actual_saved_files = sorted(
        _normalize_run_relative_path(run_dir, path_value)
        for path_value in metrics.get("files_saved", [])
    )
    expected_saved_files = sorted(spec.expected_saved_files)
    if actual_saved_files != expected_saved_files:
        raise RuntimeError(
            f"Validation for {spec.mode} expected files_saved={expected_saved_files}, got {actual_saved_files}."
        )

    for metric_name in ("simulation_time_seconds", "trial_elapsed_seconds", "path_length"):
        metric_value = float(metrics.get(metric_name, 0.0))
        if metric_value < 0.0:
            raise RuntimeError(
                f"Validation for {spec.mode} expected metrics[{metric_name!r}] to be non-negative, got {metric_value}."
            )

    if spec.mode == "LEARN_HEBB" and float(metrics.get("trial_elapsed_seconds", 0.0)) <= 0.0:
        raise RuntimeError("Validation for LEARN_HEBB expected trial_elapsed_seconds to be positive.")

    update_run_summary(
        run_dir,
        validation={
            "status": "passed",
            "suite": "base_mode_validation",
            "checked_at": datetime.now().isoformat(),
            "expected_completion_reason": spec.expected_completion_reason,
            "required_artifacts": list(spec.required_artifacts),
            "forbidden_artifacts": list(spec.forbidden_artifacts),
            "expected_saved_files": list(spec.expected_saved_files),
        },
    )

    return ValidationResult(
        mode=spec.mode,
        run_id=run_id,
        run_dir=run_dir,
        completion_reason=spec.expected_completion_reason,
        expected_artifacts=list(spec.required_artifacts),
        files_saved=actual_saved_files,
    )


def _print_smoke_suite_summary(results: list[SmokeTestResult]) -> None:
    print("Smoke suite summary:")
    for result in results:
        print(f"  PASS {result.mode}: {result.run_dir}")


def _print_session_suite_summary(results: list[SessionSmokeTestResult]) -> None:
    print("Single-session smoke suite summary:")
    for result in results:
        run_dirs = ", ".join(str(run_dir) for run_dir in result.run_dirs)
        print(f"  PASS {result.mode}: {run_dirs}")


def _print_validation_suite_summary(
    results: list[ValidationResult],
    failures: list[str],
) -> None:
    print("Base-mode validation summary:")
    for result in results:
        expected_artifacts = ", ".join(result.expected_artifacts)
        files_saved = ", ".join(result.files_saved)
        print(
            f"  PASS {result.mode}: reason={result.completion_reason}; "
            f"artifacts=[{expected_artifacts}]; files_saved=[{files_saved}]"
        )
    for failure in failures:
        print(f"  FAIL {failure}")
    print(f"  Totals: {len(results)} passed, {len(failures)} failed")


def run_smoke_test_suite() -> list[SmokeTestResult]:
    smoke_config = get_smoke_test_config()
    cleanup_stale_generated_worlds()
    world_copy = prepare_controller_world(
        smoke_config["world"],
        controller_name="multi_grid_simple",
        suffix="smoke",
    )
    source_world_name = canonical_world_name(smoke_config["world"])

    results: list[SmokeTestResult] = []
    try:
        for mode in smoke_config["modes"]:
            run_id = f"smoke_{mode.lower()}"
            run_dir = RUNS_DIR / run_id
            _reset_run_dir(run_dir)

            env = {
                "MULTI_GRID_SIMPLE_SELECTED_MODE": mode,
                "MULTI_GRID_SIMPLE_MODE_PARAMS_JSON": json.dumps(
                    _mode_overrides(mode, smoke_config, run_id)
                ),
                "MULTI_GRID_SIMPLE_AUTOMATION_CONFIG_JSON": json.dumps(
                    _automation_overrides()
                ),
                "MULTI_GRID_SIMPLE_CANONICAL_WORLD_NAME": source_world_name,
            }

            process = launch_webots(world_copy, env=env)
            _wait_for_run_completion(
                process,
                timeout_seconds=smoke_config["timeout_seconds"],
                label=f"Smoke test for {mode}",
            )

            _verify_run_artifacts(mode, run_dir)
            results.append(SmokeTestResult(mode=mode, run_id=run_id, run_dir=run_dir))

            time.sleep(1.0)
    finally:
        cleanup_generated_world(world_copy)

    return results


def run_single_session_smoke_suite() -> list[SessionSmokeTestResult]:
    smoke_config = get_smoke_test_config()
    cleanup_stale_generated_worlds()
    world_copy = prepare_controller_world(
        smoke_config["world"],
        controller_name="multi_grid_simple",
        suffix="session_smoke",
    )
    source_world_name = canonical_world_name(smoke_config["world"])

    results: list[SessionSmokeTestResult] = []
    try:
        for mode in smoke_config["modes"]:
            run_id_prefix = f"session_{mode.lower()}"
            num_trials = int(smoke_config.get("series_num_trials", 2))
            run_ids = [f"{run_id_prefix}_trial_{trial_index:03d}" for trial_index in range(1, num_trials + 1)]
            run_dirs = [RUNS_DIR / run_id for run_id in run_ids]
            for run_dir in run_dirs:
                _reset_run_dir(run_dir)

            env = {
                "MULTI_GRID_SIMPLE_SELECTED_MODE": mode,
                "MULTI_GRID_SIMPLE_MODE_PARAMS_JSON": json.dumps(
                    _mode_overrides(mode, smoke_config)
                ),
                "MULTI_GRID_SIMPLE_AUTOMATION_CONFIG_JSON": json.dumps(
                    {
                        **_automation_overrides(),
                        "quit_on_completion": True,
                    }
                ),
                "MULTI_GRID_SIMPLE_EXECUTION_CONFIG_JSON": json.dumps(
                    _series_execution_overrides(smoke_config, run_id_prefix)
                ),
                "MULTI_GRID_SIMPLE_CANONICAL_WORLD_NAME": source_world_name,
            }

            process = launch_webots(world_copy, env=env)
            _wait_for_run_completion(
                process,
                timeout_seconds=smoke_config["timeout_seconds"],
                label=f"Single-session smoke test for {mode}",
            )

            for run_dir in run_dirs:
                _verify_run_artifacts(mode, run_dir)

            results.append(SessionSmokeTestResult(mode=mode, run_ids=run_ids, run_dirs=run_dirs))
            time.sleep(1.0)
    finally:
        cleanup_generated_world(world_copy)

    return results


def run_base_mode_validation_suite() -> list[ValidationResult]:
    smoke_config = get_smoke_test_config()
    validation_modes = smoke_config.get("validation_modes", ["LEARN_HEBB", "DMTP", "EXPLOIT"])
    cleanup_stale_generated_worlds()
    world_copy = prepare_controller_world(
        smoke_config["world"],
        controller_name="multi_grid_simple",
        suffix="validation",
    )
    source_world_name = canonical_world_name(smoke_config["world"])

    results: list[ValidationResult] = []
    failures: list[str] = []
    try:
        for mode in validation_modes:
            run_id = f"validate_{mode.lower()}"
            run_dir = RUNS_DIR / run_id
            _reset_run_dir(run_dir)

            spec = _build_validation_spec(mode, smoke_config)
            env = {
                "MULTI_GRID_SIMPLE_SELECTED_MODE": mode,
                "MULTI_GRID_SIMPLE_MODE_PARAMS_JSON": json.dumps(
                    _mode_overrides(
                        mode,
                        smoke_config,
                        run_id,
                        extra_overrides=spec.mode_params_overrides,
                    )
                ),
                "MULTI_GRID_SIMPLE_AUTOMATION_CONFIG_JSON": json.dumps(
                    _automation_overrides()
                ),
                "MULTI_GRID_SIMPLE_CANONICAL_WORLD_NAME": source_world_name,
            }

            try:
                process = launch_webots(world_copy, env=env)
                _wait_for_run_completion(
                    process,
                    timeout_seconds=smoke_config["timeout_seconds"],
                    label=f"Validation for {mode}",
                )
                results.append(_validate_mode_run(spec, run_id, run_dir))
            except Exception as exc:
                if run_dir.exists():
                    update_run_summary(
                        run_dir,
                        validation={
                            "status": "failed",
                            "suite": "base_mode_validation",
                            "checked_at": datetime.now().isoformat(),
                            "failure": str(exc),
                        },
                    )
                failures.append(f"{mode}: {exc}")

            time.sleep(1.0)
    finally:
        cleanup_generated_world(world_copy)

    _print_validation_suite_summary(results, failures)

    if failures:
        raise RuntimeError(
            f"Base-mode validation suite failed for {len(failures)} mode(s): {'; '.join(failures)}"
        )

    return results


if __name__ == "__main__":
    smoke_results = run_smoke_test_suite()
    _print_smoke_suite_summary(smoke_results)

    session_results = run_single_session_smoke_suite()
    _print_session_suite_summary(session_results)

    run_base_mode_validation_suite()
