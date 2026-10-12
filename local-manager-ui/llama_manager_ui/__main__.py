"""Entry point: python3 -m llama_manager_ui"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llama-manager-ui",
        description="GTK3 front-end for local-manager: launch llama.cpp servers and run builds.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.json")
    parser.add_argument("--version", action="version", version="llama-manager-ui 1.0.0")
    args = parser.parse_args(argv)

    try:
        from .gtk_ui import run
    except ImportError as exc:
        print(f"ERROR: GTK3/PyGObject is required: {exc}", file=sys.stderr)
        print("Install with: sudo apt install python3-gi gir1.2-gtk-3.0", file=sys.stderr)
        return 1

    return run(config_path=args.config)


if __name__ == "__main__":
    raise SystemExit(main())
