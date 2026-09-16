"""Config load/save for local-manager."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.json"
EXAMPLE_CONFIG_PATH = ROOT / "config.example.json"

DEFAULTS: dict[str, Any] = {
    "models_dir": "/home/tjhexa/development/llama.cpp/models",
    "bin_dir": "/home/tjhexa/development/llama.cpp/llama.cpp/build/bin",
    "server_bin": "llama-server",
    "cli_bin": "llama-completion",
    "default_port": 9898,
    "default_host": "127.0.0.1",
    "default_ngl": 99,
    "default_ctx": 0,
    "open_browser": True,
    "api_key": "",
    "alias": "",
    "extra_server_args": [],
    "extra_cli_args": [],
    "skip_mmproj": False,
    "history_file": "history.json",
}


def ensure_config(path: Path | None = None) -> Path:
    cfg_path = path or DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        src = EXAMPLE_CONFIG_PATH if EXAMPLE_CONFIG_PATH.exists() else None
        data = (
            json.loads(src.read_text(encoding="utf-8"))
            if src
            else deepcopy(DEFAULTS)
        )
        cfg_path.write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )
    return cfg_path


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = ensure_config(path)
    data = deepcopy(DEFAULTS)
    loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
    data.update(loaded)
    return data


def history_path(cfg: dict[str, Any]) -> Path:
    name = cfg.get("history_file") or "history.json"
    p = Path(name)
    if not p.is_absolute():
        p = ROOT / p
    return p


def load_history(cfg: dict[str, Any]) -> dict[str, Any]:
    path = history_path(cfg)
    if not path.exists():
        return {"last_model": "", "last_mode": "web", "launches": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"last_model": "", "last_mode": "web", "launches": []}


def save_history(cfg: dict[str, Any], hist: dict[str, Any]) -> None:
    path = history_path(cfg)
    path.write_text(json.dumps(hist, indent=2) + "\n", encoding="utf-8")
