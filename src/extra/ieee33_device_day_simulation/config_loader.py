from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CONFIG_NAMES = (
    "paths",
    "original_model",
    "simulation",
    "population",
    "source_device_map",
    "zones",
    "network",
    "device_constraints",
    "user_behavior",
    "control",
    "experiment",
    "outputs",
)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or Path(__file__).parent / "configs" / "default.yaml")
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as fh:
        root = yaml.safe_load(fh) or {}
    result: dict[str, Any] = {"config_dir": str(config_path.parent)}
    for name in CONFIG_NAMES:
        relative = root.get(name)
        if not relative:
            raise ValueError(f"default config does not reference '{name}'")
        file_path = config_path.parent / relative
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        with file_path.open("r", encoding="utf-8") as fh:
            result[name] = yaml.safe_load(fh) or {}
    result["default_file"] = str(config_path)
    return result


def resolve_workspace_path(config: dict[str, Any], key: str) -> Path:
    value = config["paths"][key]
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path
