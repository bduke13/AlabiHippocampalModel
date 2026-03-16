import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

try:
    from config import get_launch_config
except ImportError:  # Support package import from repo root.
    from .config import get_launch_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORLD_ROOT = PROJECT_ROOT / "webots" / "worlds"
GENERATED_WORLD_ROOT = WORLD_ROOT


def resolve_webots_executable(explicit_path: Optional[str] = None) -> Path:
    if explicit_path:
        executable = Path(explicit_path)
        if executable.exists():
            return executable
        raise FileNotFoundError(f"Webots executable not found: {explicit_path}")

    env_executable = os.environ.get("WEBOTS_EXECUTABLE")
    if env_executable:
        executable = Path(env_executable)
        if executable.exists():
            return executable
        raise FileNotFoundError(f"WEBOTS_EXECUTABLE points to a missing file: {env_executable}")

    webots_home = os.environ.get("WEBOTS_HOME")
    if webots_home:
        candidates = [
            Path(webots_home) / "msys64" / "mingw64" / "bin" / "webots.exe",
            Path(webots_home) / "webots.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

    which_path = shutil.which("webots")
    if which_path:
        return Path(which_path)

    raise FileNotFoundError(
        "Could not resolve webots executable. Set WEBOTS_EXECUTABLE, set WEBOTS_HOME, "
        "or ensure `webots` is available on PATH."
    )


def resolve_world_path(world_name_or_path: str) -> Path:
    candidate = Path(world_name_or_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate.resolve()

    world_path = WORLD_ROOT / world_name_or_path
    if world_path.exists():
        return world_path

    raise FileNotFoundError(f"World file not found: {world_name_or_path}")


def prepare_controller_world(
    source_world: str | Path,
    *,
    controller_name: str = "multi_grid_simple",
    suffix: str = "automation",
) -> Path:
    source_path = resolve_world_path(str(source_world))
    GENERATED_WORLD_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = GENERATED_WORLD_ROOT / f"{source_path.stem}_{suffix}.wbt"

    world_text = source_path.read_text(encoding="utf-8")
    updated_text, replacements = re.subn(
        r'controller\s+"[^"]+"',
        f'controller "{controller_name}"',
        world_text,
        count=1,
    )
    if replacements == 0:
        raise ValueError(f"No controller field found in world file: {source_path}")

    output_path.write_text(updated_text, encoding="utf-8")
    return output_path


def build_webots_command(
    world_path: str | Path,
    *,
    webots_executable: Optional[str] = None,
    mode: str = "fast",
    batch: bool = True,
    no_rendering: bool = True,
    stdout: bool = True,
    stderr: bool = True,
    minimize: bool = False,
    port: Optional[int] = None,
    heartbeat: Optional[int] = None,
) -> list[str]:
    executable = resolve_webots_executable(webots_executable)
    command = [str(executable)]

    if mode:
        command.append(f"--mode={mode}")
    if batch:
        command.append("--batch")
    if no_rendering:
        command.append("--no-rendering")
    if stdout:
        command.append("--stdout")
    if stderr:
        command.append("--stderr")
    if minimize:
        command.append("--minimize")
    if port is not None:
        command.append(f"--port={port}")
    if heartbeat is not None:
        command.append(f"--heartbeat={heartbeat}")

    command.append(str(resolve_world_path(str(world_path))))
    return command


def launch_webots(
    world_path: str | Path,
    *,
    env: Optional[dict] = None,
    **launch_kwargs,
) -> subprocess.Popen:
    command = build_webots_command(world_path, **launch_kwargs)
    child_env = os.environ.copy()
    if env:
        child_env.update(env)

    return subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        env=child_env,
    )


def launch_from_config(
    *,
    controller_name: str = "multi_grid_simple",
    env: Optional[dict] = None,
) -> subprocess.Popen:
    launch_config = get_launch_config()
    world_path = prepare_controller_world(
        launch_config["world"],
        controller_name=controller_name,
    )
    return launch_webots(
        world_path,
        env=env,
        webots_executable=launch_config.get("webots_executable"),
        mode=launch_config.get("mode", "fast"),
        batch=bool(launch_config.get("batch", True)),
        no_rendering=bool(launch_config.get("no_rendering", True)),
        stdout=bool(launch_config.get("stdout", True)),
        stderr=bool(launch_config.get("stderr", True)),
        minimize=bool(launch_config.get("minimize", False)),
        port=launch_config.get("port"),
        heartbeat=launch_config.get("heartbeat"),
    )
