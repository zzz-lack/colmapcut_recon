"""Load project, scene, and external-tool YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and report malformed or missing configuration clearly."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Top-level YAML value must be a mapping: {path}")
    return value


def resolve_project_path(value: str | Path, *, root: Path = PROJECT_ROOT) -> Path:
    """Resolve a path relative to the project root, never the caller's cwd."""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_tool(tool_name: str, tools_config: Path) -> dict[str, Any]:
    """Return one configured external tool mapping."""

    tools = load_yaml(tools_config)
    value = tools.get(tool_name)
    if not isinstance(value, dict):
        raise TypeError(f"Tool '{tool_name}' is not configured in {tools_config}")
    return value
