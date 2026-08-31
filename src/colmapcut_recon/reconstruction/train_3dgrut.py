"""Launch the external 3DGRUT trainer with project-owned inputs and outputs."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from colmapcut_recon.common.config import (
    PROJECT_ROOT,
    load_tool,
    load_yaml,
    resolve_project_path,
)
from colmapcut_recon.common.subprocess_utils import format_command, run_command

BACKGROUND_NAMES = {
    (0.0, 0.0, 0.0): "black",
    (1.0, 1.0, 1.0): "white",
}


def _python_environment(python: Path) -> tuple[dict[str, str], dict[str, str | None]]:
    """Build an execution environment that exposes venv-provided CLI tools."""

    environment = os.environ.copy()
    path_prefix = str(python.parent)
    existing_path = environment.get("PATH", "")
    environment["PATH"] = (
        path_prefix if not existing_path else path_prefix + os.pathsep + existing_path
    )
    virtual_environment = python.parent.parent
    virtual_environment_text: str | None = None
    if (virtual_environment / "pyvenv.cfg").is_file():
        virtual_environment_text = str(virtual_environment)
        environment["VIRTUAL_ENV"] = virtual_environment_text
    return environment, {
        "path_prepend": path_prefix,
        "virtual_env": virtual_environment_text,
    }


def _background_name(value: Any) -> str:
    if isinstance(value, str) and value in {"black", "white", "random"}:
        return value
    if isinstance(value, list) and len(value) == 3:
        key = tuple(float(component) for component in value)
        if key in BACKGROUND_NAMES:
            return BACKGROUND_NAMES[key]
    raise ValueError(
        "3DGRUT background_color must be black, white, random, [0,0,0], or [1,1,1]"
    )


def build_3dgrut_command(
    *,
    python: Path,
    repository: Path,
    dataset_root: Path,
    run_dir: Path,
    config: dict[str, Any],
) -> list[str]:
    """Build the Hydra command used by the installed 3DGRUT repository."""

    app_config = str(config.get("app_config", "apps/colmap_3dgut.yaml"))
    command = [
        str(python),
        str(repository / "train.py"),
        "--config-name",
        app_config,
        f"path={dataset_root}",
        f"out_dir={run_dir}",
        f"model.background.color={_background_name(config.get('background_color', 'white'))}",
    ]
    experiment_name = str(config.get("experiment_name", "")).strip()
    if experiment_name:
        command.append(f"experiment_name={experiment_name}")
    downsample = int(config.get("downsample_factor", 1))
    command.append(f"dataset.downsample_factor={downsample}")
    if "iterations" in config:
        command.append(f"n_iterations={int(config['iterations'])}")
    if "num_workers" in config:
        command.append(f"num_workers={int(config['num_workers'])}")
    if bool(config.get("export_ply", True)):
        command.append("export_ply.enabled=true")
    command.extend(str(value) for value in config.get("extra_args", []))
    return command


def train_3dgrut(
    *,
    python: Path,
    repository: Path,
    dataset_root: Path,
    run_dir: Path,
    config: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, object]:
    """Validate, record, and optionally execute external 3DGRUT training."""

    # Keep the virtual-environment entry path intact. Resolving a symlink such
    # as ``.venv/bin/python`` to its base interpreter bypasses ``pyvenv.cfg``
    # and silently drops the environment's installed packages.
    python = python.expanduser().absolute()
    repository = repository.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    if not python.is_file():
        raise FileNotFoundError(f"3DGRUT Python executable does not exist: {python}")
    if not (repository / "train.py").is_file():
        raise FileNotFoundError(f"3DGRUT train.py does not exist under: {repository}")
    required = [dataset_root / "images", dataset_root / "sparse" / "0"]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"3DGRUT dataset is incomplete; missing: {missing}")

    command = build_3dgrut_command(
        python=python,
        repository=repository,
        dataset_root=dataset_root,
        run_dir=run_dir,
        config=config,
    )
    runtime_environment, environment_record = _python_environment(python)
    manifest: dict[str, object] = {
        "adapter": "colmapcut_recon.3dgrut",
        "dry_run": dry_run,
        "repository": str(repository),
        "python": str(python),
        "dataset": str(dataset_root),
        "output_root": str(run_dir),
        "command": command,
        "command_text": format_command(command),
        "environment": environment_record,
    }
    if dry_run:
        return manifest

    run_dir.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for path in run_dir.iterdir() if path.is_dir()}
    invocation_path = run_dir / "last_invocation.json"
    run_command(
        command,
        cwd=repository,
        env=runtime_environment,
        record_path=run_dir / "last_process.json",
    )
    after = {path.resolve() for path in run_dir.iterdir() if path.is_dir()}
    manifest["new_run_directories"] = [str(path) for path in sorted(after - before)]
    manifest["invocation_manifest"] = str(invocation_path)
    invocation_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-config",
        type=Path,
        default=PROJECT_ROOT / "configs/scenes/plant_001.yaml",
    )
    parser.add_argument(
        "--tools-config", type=Path, default=PROJECT_ROOT / "configs/tools.local.yaml"
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=PROJECT_ROOT / "configs/training/3dgrut.yaml",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scene = load_yaml(args.scene_config)
    config = load_yaml(args.training_config)
    tool = load_tool("threedgrut", args.tools_config)
    scene_id = str(scene["scene_id"])
    data_root = resolve_project_path(scene["data_root"])
    dataset_root = args.dataset_root or data_root / str(
        config.get("dataset_stage", "07_datasets/3dgrut")
    )
    run_root = resolve_project_path(config.get("run_root", "runs"))
    run_dir = args.run_dir or run_root / scene_id / "3dgrut"
    manifest = train_3dgrut(
        python=args.python or Path(tool["python"]),
        repository=args.repository or Path(tool["repository"]),
        dataset_root=dataset_root,
        run_dir=run_dir,
        config=config,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
