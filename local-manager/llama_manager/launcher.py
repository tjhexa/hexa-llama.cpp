"""Build and run llama-server / llama-completion commands."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import ModelEntry
from .system_info import port_in_use


@dataclass
class LaunchOptions:
    mode: str  # web | cli
    model: ModelEntry
    bin_dir: Path
    server_bin: str = "llama-server"
    # llama-completion supports -cnv; newer llama-cli does not.
    cli_bin: str = "llama-completion"
    host: str = "127.0.0.1"
    port: int = 9898
    ngl: int = 99
    ctx: int = 0
    alias: str = ""
    api_key: str = ""
    open_browser: bool = True
    use_mmproj: bool = True
    extra_args: list[str] = field(default_factory=list)
    cli_prompt: str = ""
    dry_run: bool = False


def resolve_bin(bin_dir: Path, name: str) -> Path:
    p = Path(name)
    if p.is_absolute():
        return p
    cand = bin_dir / name
    if cand.exists():
        return cand
    which = shutil.which(name)
    if which:
        return Path(which)
    return cand


def _cli_help_text(bin_path: Path) -> str:
    try:
        proc = subprocess.run(
            [str(bin_path), "-h"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


def _cli_supports_cnv(bin_path: Path) -> bool:
    """True if binary help mentions -cnv (llama-completion does; llama-cli may not)."""
    text = _cli_help_text(bin_path)
    if text:
        return "-cnv" in text or "--conversation" in text
    return "completion" in bin_path.name


def _cli_supports_mmproj(bin_path: Path) -> bool:
    """True if binary accepts -mm/--mmproj FILE (llama-cli/server do; completion does not)."""
    text = _cli_help_text(bin_path)
    if text:
        # Require the FILE form, not just --no-mmproj from -hf help text.
        return "-mm," in text or "-mm, " in text or "--mmproj FILE" in text
    return "completion" not in bin_path.name


def _pick_cli_bin(opts: LaunchOptions, want_mmproj: bool) -> Path:
    """
    Choose a working CLI binary:
    - text chat: prefer llama-completion (-cnv)
    - with mmproj: prefer llama-cli (supports -mm; no -cnv needed)
    """
    preferred: list[str] = []
    if want_mmproj:
        preferred.extend(["llama-cli", "llama-mtmd-cli", opts.cli_bin, "llama-completion"])
    else:
        preferred.extend([opts.cli_bin, "llama-completion", "llama-cli"])

    seen: set[str] = set()
    candidates: list[Path] = []
    for name in preferred:
        if not name or name in seen:
            continue
        seen.add(name)
        candidates.append(resolve_bin(opts.bin_dir, name))

    for cand in candidates:
        if not cand.exists():
            continue
        if want_mmproj and not _cli_supports_mmproj(cand):
            continue
        return cand

    # Last resort: first existing path even if flags may not match.
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def build_command(opts: LaunchOptions) -> list[str]:
    model_path = str(opts.model.path)
    want_mmproj = bool(
        opts.use_mmproj and opts.model.mmproj and opts.model.mmproj.exists()
    )

    if opts.mode == "web":
        bin_path = resolve_bin(opts.bin_dir, opts.server_bin)
        cmd = [
            str(bin_path),
            "-m",
            model_path,
            "-ngl",
            str(opts.ngl),
            "--host",
            opts.host,
            "--port",
            str(opts.port),
        ]
        if opts.ctx and opts.ctx > 0:
            cmd.extend(["-c", str(opts.ctx)])
        alias = opts.alias.strip() or opts.model.path.stem
        cmd.extend(["-a", alias])
        if opts.api_key:
            cmd.extend(["--api-key", opts.api_key])
        if want_mmproj:
            # llama-server accepts long form
            cmd.extend(["--mmproj", str(opts.model.mmproj)])
        cmd.extend(opts.extra_args)
        return cmd

    bin_path = _pick_cli_bin(opts, want_mmproj=want_mmproj)
    cmd = [
        str(bin_path),
        "-m",
        model_path,
        "-ngl",
        str(opts.ngl),
    ]
    if opts.ctx and opts.ctx > 0:
        cmd.extend(["-c", str(opts.ctx)])

    if want_mmproj:
        if _cli_supports_mmproj(bin_path):
            # Short -mm is accepted by llama-cli / server / mtmd-cli
            cmd.extend(["-mm", str(opts.model.mmproj)])
        else:
            print(
                "WARN: selected CLI binary has no mmproj support;"
                f" skipping projector ({bin_path.name})."
            )

    supports_cnv = _cli_supports_cnv(bin_path)
    # llama-completion disables jinja by default; custom templates (e.g. Gemma 4)
    # throw: "this custom template is not supported, try using --jinja"
    if "--jinja" not in opts.extra_args and "--no-jinja" not in opts.extra_args:
        cmd.append("--jinja")

    if opts.cli_prompt:
        if supports_cnv:
            cmd.extend(["-no-cnv", "-p", opts.cli_prompt])
        else:
            cmd.extend(["-st", "-p", opts.cli_prompt])
    else:
        if supports_cnv:
            cmd.append("-cnv")
        # llama-cli: conversation is auto when chat template exists

    cmd.extend(opts.extra_args)
    return cmd


def command_to_shell(cmd: list[str]) -> str:
    return " ".join(_shell_quote(c) for c in cmd)


def _shell_quote(s: str) -> str:
    if re_needs_quote(s):
        return "'" + s.replace("'", "'\"'\"'") + "'"
    return s


def re_needs_quote(s: str) -> bool:
    return any(c in s for c in ' \t\n"$&\'();<>\\|')


def run_launch(opts: LaunchOptions) -> int:
    cmd = build_command(opts)
    print()
    print("Launch command:")
    print(" ", command_to_shell(cmd))
    print()

    bin_path = Path(cmd[0])
    if not bin_path.exists():
        print(f"ERROR: binary not found: {bin_path}")
        print("Build llama.cpp first, or fix bin_dir in config.json")
        return 1

    if opts.dry_run:
        print("Dry-run only; not starting.")
        return 0

    if opts.mode == "web":
        return _run_web(opts, cmd)
    return _run_cli(cmd)


def _run_cli(cmd: list[str]) -> int:
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 130


def _run_web(opts: LaunchOptions, cmd: list[str]) -> int:
    url = f"http://127.0.0.1:{opts.port}"
    if opts.host not in ("0.0.0.0", "127.0.0.1", "localhost"):
        url = f"http://{opts.host}:{opts.port}"

    if port_in_use(opts.host, opts.port):
        print(f"Port {opts.port} already in use.")
        print(f"Opening existing server UI: {url}")
        if opts.open_browser:
            _open_browser(url)
        return 0

    if opts.open_browser:
        t = threading.Thread(
            target=_wait_and_open_browser,
            args=(url, 180),
            daemon=True,
        )
        t.start()

    print(f"Starting web server on {opts.host}:{opts.port}")
    print(f"UI / OpenAI base: {url}  and  {url}/v1")
    try:
        os.execvp(cmd[0], cmd)
    except OSError as exc:
        print(f"ERROR: failed to exec: {exc}")
        return 1
    return 0


def _wait_and_open_browser(url: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _http_up(url):
            _open_browser(url)
            return
        time.sleep(1)


def _http_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1) as resp:
            return 200 <= getattr(resp, "status", 200) < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _open_browser(url: str) -> None:
    opener = shutil.which("xdg-open")
    if not opener:
        return
    try:
        subprocess.Popen(
            [opener, url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


def record_launch(hist: dict[str, Any], opts: LaunchOptions) -> dict[str, Any]:
    hist["last_model"] = str(opts.model.path)
    hist["last_mode"] = opts.mode
    launches = hist.get("launches") or []
    launches.insert(
        0,
        {
            "model": str(opts.model.path),
            "mode": opts.mode,
            "port": opts.port,
            "ngl": opts.ngl,
            "ctx": opts.ctx,
        },
    )
    hist["launches"] = launches[:20]
    return hist
