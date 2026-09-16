"""llama local-manager entrypoint.

Interactive launcher for llama.cpp web server or CLI inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import load_config, load_history, save_history
from .launcher import LaunchOptions, build_command, command_to_shell, record_launch, run_launch
from .models import discover_models, find_by_path
from .system_info import collect_system_info
from .ui import (
    print_header,
    print_model_details,
    print_model_list,
    print_system,
    prompt_choice,
    prompt_int,
    prompt_str,
    prompt_yn,
    select_model_index,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llama-manager",
        description=(
            "Scan local GGUF models, show RAM/VRAM + metadata, "
            "then launch llama-server (web) or llama-completion (cli)."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"llama-manager {__version__}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.json (default: local-manager/config.json)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List models with details and exit",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model path or filename (skip interactive select)",
    )
    parser.add_argument(
        "--mode",
        choices=("web", "cli"),
        default="",
        help="Launch mode: web=llama-server, cli=llama-completion",
    )
    parser.add_argument("--host", default="", help="Server bind host")
    parser.add_argument("--port", type=int, default=-1, help="Server port")
    parser.add_argument(
        "--ngl",
        type=int,
        default=-1,
        help="GPU layers (-ngl). 0=CPU",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=-1,
        help="Context size (-c). 0=model/default",
    )
    parser.add_argument("--alias", default="", help="API model alias (-a)")
    parser.add_argument("--api-key", default="", help="Optional API key")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open browser for web mode",
    )
    parser.add_argument(
        "--no-mmproj",
        action="store_true",
        help="Do not attach sibling mmproj",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="CLI one-shot prompt (implies non-conversation if set)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command only",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip final confirmation",
    )

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    hist = load_history(cfg)
    sysinfo = collect_system_info()

    models_dir = Path(cfg["models_dir"]).expanduser()
    bin_dir = Path(cfg["bin_dir"]).expanduser()

    print_header(f"llama-manager {__version__}")
    print(f"models_dir : {models_dir}")
    print(f"bin_dir    : {bin_dir}")
    print_system(sysinfo)

    print("\nScanning models...")
    entries = discover_models(models_dir)
    print(f"Found {len(entries)} model(s)\n")
    print_model_list(entries, last_model=hist.get("last_model", ""))

    if args.list:
        if entries:
            print()
            for e in entries:
                print_model_details(
                    e,
                    sysinfo,
                    ctx_for_hint=cfg.get("default_ctx") or None,
                )
        return 0

    # Resolve model
    if args.model:
        entry = find_by_path(entries, args.model)
        if entry is None:
            # Allow absolute path not under models_dir
            p = Path(args.model).expanduser()
            if p.is_file():
                from .gguf_meta import read_gguf_info
                from .models import ModelEntry

                entry = ModelEntry(info=read_gguf_info(p))
            else:
                print(f"ERROR: model not found: {args.model}")
                return 1
        model_idx = None
    else:
        model_idx = select_model_index(
            entries, last_model=hist.get("last_model", "")
        )
        entry = entries[model_idx - 1]

    print_model_details(
        entry,
        sysinfo,
        ctx_for_hint=cfg.get("default_ctx") or entry.info.context_length,
    )

    # Mode
    mode = args.mode
    if not mode:
        default_mode = hist.get("last_mode") or "web"
        print()
        print("Launch mode:")
        print("  [1] web  - llama-server (browser UI + OpenAI /v1 API)")
        print("  [2] cli  - llama-completion (terminal chat / prompt)")
        choice = prompt_choice(
            "Choose 1 or 2",
            {"1", "2", "web", "cli"},
            "1" if default_mode == "web" else "2",
        )
        mode = "web" if choice in ("1", "web") else "cli"

    host = args.host or str(cfg.get("default_host") or "127.0.0.1")
    port = args.port if args.port >= 0 else int(cfg.get("default_port") or 9898)
    ngl = args.ngl if args.ngl >= 0 else int(cfg.get("default_ngl") or 99)
    ctx = args.ctx if args.ctx >= 0 else int(cfg.get("default_ctx") or 0)
    alias = args.alias or str(cfg.get("alias") or "")
    api_key = args.api_key or str(cfg.get("api_key") or "")
    open_browser = bool(cfg.get("open_browser", True)) and not args.no_browser
    use_mmproj = not args.no_mmproj and not bool(cfg.get("skip_mmproj", False))

    # Interactive tuning unless fully specified non-interactively with -y
    interactive = not args.yes or not args.mode or not args.model
    if interactive and sys.stdin.isatty():
        print()
        print("Runtime options (press Enter to keep defaults):")
        ngl = prompt_int("GPU layers -ngl (0=CPU)", ngl, minimum=0)
        max_ctx = entry.info.context_length
        ctx = prompt_int(
            "Context -c (0=auto/model default"
            + (f", model max={max_ctx}" if max_ctx else "")
            + ")",
            ctx,
            minimum=0,
        )
        if mode == "web":
            host = prompt_str("Host", host)
            port = prompt_int("Port", port, minimum=1, maximum=65535)
            alias = prompt_str(
                "API alias (-a)",
                alias or entry.path.stem,
            )
            if not api_key:
                api_key = prompt_str("API key (empty=disabled)", "")
            open_browser = prompt_yn("Open browser when ready", open_browser)
        else:
            if not args.prompt:
                if prompt_yn("One-shot prompt instead of interactive chat", False):
                    args.prompt = prompt_str("Prompt text", "Hello")
        if entry.mmproj and use_mmproj:
            use_mmproj = prompt_yn(
                f"Attach mmproj ({entry.mmproj.name})", True
            )

    extra = list(
        cfg.get("extra_server_args")
        if mode == "web"
        else cfg.get("extra_cli_args")
        or []
    )

    opts = LaunchOptions(
        mode=mode,
        model=entry,
        bin_dir=bin_dir,
        server_bin=str(cfg.get("server_bin") or "llama-server"),
        cli_bin=str(cfg.get("cli_bin") or "llama-completion"),
        host=host,
        port=port,
        ngl=ngl,
        ctx=ctx,
        alias=alias,
        api_key=api_key,
        open_browser=open_browser,
        use_mmproj=use_mmproj,
        extra_args=[str(x) for x in extra],
        cli_prompt=args.prompt,
        dry_run=args.dry_run,
    )

    print_header("Ready to launch")
    print(f"  mode   : {opts.mode}")
    print(f"  model  : {opts.model.path}")
    print(f"  ngl    : {opts.ngl}")
    print(f"  ctx    : {opts.ctx if opts.ctx else 'auto'}")
    if opts.mode == "web":
        print(f"  listen : {opts.host}:{opts.port}")
        print(f"  alias  : {opts.alias or opts.model.path.stem}")
        print(f"  openai : http://127.0.0.1:{opts.port}/v1")
    print(f"  cmd    : {command_to_shell(build_command(opts))}")

    if not args.yes and sys.stdin.isatty():
        if not prompt_yn("Start now", True):
            print("Aborted.")
            return 0

    hist = record_launch(hist, opts)
    save_history(cfg, hist)
    return run_launch(opts)


if __name__ == "__main__":
    raise SystemExit(main())
