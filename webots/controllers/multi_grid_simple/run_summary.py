import json
from datetime import datetime
from pathlib import Path
from typing import Optional


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


def _to_relative(path_value: str | Path, root: Path) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        return path.as_posix()
    return path.resolve().relative_to(root.resolve()).as_posix()


def _list_relative_files(path: Path, root: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(
        file_path.resolve().relative_to(root.resolve()).as_posix()
        for file_path in path.rglob("*")
        if file_path.is_file()
    )


def _build_review_checks(summary: dict) -> list[dict]:
    validation = summary.get("validation", {})
    plots = summary.get("artifacts", {}).get("plots", [])
    files_saved = summary.get("artifacts", {}).get("files_saved", [])
    checks = [
        {
            "id": "trial_completed",
            "label": "Trial completed",
            "status": "pass" if summary.get("status") == "completed" else "fail",
        },
        {
            "id": "artifacts_saved",
            "label": "Expected outputs were saved",
            "status": "pass" if len(files_saved) > 0 else "warn",
        },
        {
            "id": "plots_generated",
            "label": "Verification plots generated",
            "status": "pass" if len(plots) > 0 else "warn",
        },
        {
            "id": "validation_passed",
            "label": "Validation passed",
            "status": (
                "pass"
                if validation.get("status") == "passed"
                else "warn"
                if not validation
                else "fail"
            ),
        },
    ]
    return checks


def _build_review_markdown(summary: dict) -> str:
    artifacts = summary.get("artifacts", {})
    validation = summary.get("validation", {})
    checks = summary.get("review_checks", [])
    lines = [
        "# Run Review",
        "",
        f"- Run ID: `{summary.get('run_id')}`",
        f"- Mode: `{summary.get('mode')}`",
        f"- World: `{summary.get('world_name')}`",
        f"- Status: `{summary.get('status')}`",
        f"- Completion reason: `{summary.get('completion_reason')}`",
        f"- Trial elapsed seconds: `{summary.get('trial_elapsed_seconds')}`",
        f"- Path length: `{summary.get('path_length')}`",
        "",
        "## Review Checks",
    ]
    for check in checks:
        lines.append(f"- `{check['status']}` {check['label']}")

    lines.extend(
        [
            "",
            "## Saved Files",
        ]
    )
    for relative_path in artifacts.get("files_saved", []):
        lines.append(f"- `{relative_path}`")
    if not artifacts.get("files_saved"):
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Plots",
        ]
    )
    for relative_path in artifacts.get("plots", []):
        lines.append(f"- `{relative_path}`")
    if not artifacts.get("plots"):
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Validation",
            f"- Status: `{validation.get('status', 'not_run')}`",
        ]
    )
    if validation.get("suite"):
        lines.append(f"- Suite: `{validation['suite']}`")
    if validation.get("checked_at"):
        lines.append(f"- Checked at: `{validation['checked_at']}`")
    if validation.get("failure"):
        lines.append(f"- Failure: `{validation['failure']}`")

    return "\n".join(lines) + "\n"


def update_run_summary(
    run_dir: Path,
    *,
    visualization_dir: Optional[Path] = None,
    validation: Optional[dict] = None,
    review_notes: Optional[dict] = None,
) -> dict:
    run_dir = run_dir.resolve()
    controller_dir = run_dir.parents[1]
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    summary_path = run_dir / "summary.json"
    review_path = run_dir / "review.md"

    existing_summary = _load_json(summary_path)
    config = _load_json(config_path)
    metrics = _load_json(metrics_path)

    if visualization_dir is None:
        visualization_dir = controller_dir / "visualizations" / run_dir.name

    files_saved = [
        _to_relative(path_value, run_dir)
        for path_value in metrics.get("files_saved", [])
    ]

    summary = {
        "run_id": config.get("run_id") or metrics.get("run_id") or run_dir.name,
        "mode": config.get("mode") or metrics.get("mode"),
        "world_name": config.get("world_name") or metrics.get("world_name"),
        "status": metrics.get("status"),
        "completion_reason": metrics.get("completion_reason"),
        "simulation_time_seconds": metrics.get("simulation_time_seconds"),
        "trial_elapsed_seconds": metrics.get("trial_elapsed_seconds"),
        "path_length": metrics.get("path_length"),
        "step_count": metrics.get("step_count"),
        "config": config,
        "artifacts": {
            "files_saved": sorted(files_saved),
            "network_files": _list_relative_files(run_dir / "networks", run_dir),
            "hmap_files": _list_relative_files(run_dir / "hmaps", run_dir),
            "plots": _list_relative_files(visualization_dir, controller_dir),
        },
        "validation": existing_summary.get("validation", {}),
        "review_notes": existing_summary.get("review_notes", {}),
        "updated_at": datetime.now().isoformat(),
    }

    if validation is not None:
        summary["validation"] = validation
    if review_notes is not None:
        summary["review_notes"] = review_notes

    summary["review_checks"] = _build_review_checks(summary)
    summary["ready_for_review"] = all(
        check["status"] != "fail" for check in summary["review_checks"]
    )

    _write_json(summary_path, summary)
    review_path.write_text(_build_review_markdown(summary), encoding="utf-8")
    return summary
