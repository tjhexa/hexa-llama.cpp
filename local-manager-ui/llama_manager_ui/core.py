"""Core helpers for local-manager-ui.

Self-contained: config, model discovery, GGUF reading, command building,
subprocess streaming and process control. No dependency on llama_manager.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent           # local-manager-ui/
REPO_ROOT = ROOT.parent                                  # llama.cpp/
LOCAL_MANAGER_DIR = REPO_ROOT / "local-manager"
CONFIG_PATH = ROOT / "config.json"
EXAMPLE_CONFIG_PATH = ROOT / "config.example.json"
HISTORY_PATH = ROOT / "history.json"

DEFAULT_REPO_CONFIG = LOCAL_MANAGER_DIR / "config.json"
DEFAULT_REPO_HISTORY = LOCAL_MANAGER_DIR / "history.json"
BUILD_SCRIPT = LOCAL_MANAGER_DIR / "pull-merge-build.sh"

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


# --------------------------------------------------------------------------
# config + history
# --------------------------------------------------------------------------

def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from start (or this file) until a dir containing .git is found."""
    cur = (start or ROOT).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / ".git").exists():
            return candidate
    return REPO_ROOT


def _merge(loaded: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    out.update(loaded)
    return out


def load_config(path: Path | None = None) -> tuple[dict[str, Any], Path]:
    """Load UI config, falling back to local-manager/config.json then defaults."""
    cfg_path = path or CONFIG_PATH
    if cfg_path.exists():
        try:
            return _merge(json.loads(cfg_path.read_text(encoding="utf-8")), {}), cfg_path
        except json.JSONDecodeError:
            pass
    if DEFAULT_REPO_CONFIG.exists():
        try:
            return _merge(json.loads(DEFAULT_REPO_CONFIG.read_text(encoding="utf-8")), {}), cfg_path
        except json.JSONDecodeError:
            pass
    repo = find_repo_root()
    return {
        "models_dir": str(repo.parent / "models"),
        "bin_dir": str(repo / "build" / "bin"),
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
    }, cfg_path


def save_config(cfg: dict[str, Any], path: Path | None = None) -> Path:
    cfg_path = path or CONFIG_PATH
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return cfg_path


def load_history() -> dict[str, Any]:
    for path in (HISTORY_PATH, DEFAULT_REPO_HISTORY):
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return {"last_model": "", "last_mode": "web", "launches": []}


def save_history(hist: dict[str, Any], path: Path | None = None) -> Path:
    hist_path = path or HISTORY_PATH
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps(hist, indent=2) + "\n", encoding="utf-8")
    return hist_path


def record_launch(hist: dict[str, Any], opts: "LaunchOptions") -> dict[str, Any]:
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
            "ts": int(time.time()),
        },
    )
    hist["launches"] = launches[:30]
    return hist


# --------------------------------------------------------------------------
# GGUF reading (minimal, stdlib only)
# --------------------------------------------------------------------------

GGUF_TYPE = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4),
    5: ("i", 4), 6: ("f", 4), 7: ("?", 1), 8: (None, None), 9: (None, None),
    10: ("Q", 8), 11: ("q", 8), 12: ("d", 8),
}

QUANT_TAGS = (
    "MXFP4", "IQ4_NL", "IQ4_XS", "IQ3_XXS", "IQ3_M", "IQ2_XXS", "IQ2_M",
    "IQ1_M", "IQ1_S", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q4_K_M",
    "Q4_K_S", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K", "Q1_0", "BF16", "F16", "F32",
)


@dataclass
class ModelInfo:
    path: Path
    size_bytes: int = 0
    architecture: str = ""
    name: str = ""
    size_label: str = ""
    file_type: Any = None
    context_length: int | None = None
    original_context_length: int | None = None
    embedding_length: int | None = None
    block_count: int | None = None
    rope_freq_base: float | None = None
    rope_scaling_type: str = ""
    rope_scaling_factor: float | None = None
    expert_count: int | None = None
    expert_used_count: int | None = None
    is_mmproj: bool = False
    error: str = ""

    @property
    def size_human(self) -> str:
        return human_bytes(self.size_bytes)

    @property
    def display_name(self) -> str:
        return self.name or self.path.stem

    @property
    def quant(self) -> str:
        upper = self.path.name.upper()
        for tag in QUANT_TAGS:
            if tag in upper:
                return tag
        if self.file_type is not None:
            return f"type={self.file_type}"
        return "?"


@dataclass
class ModelRef:
    """Lightweight model handle; metadata is read on demand."""

    path: Path
    mmproj: Path | None = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem


def _as_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_gguf_info(path: Path) -> ModelInfo:
    info = ModelInfo(path=path)
    try:
        info.size_bytes = path.stat().st_size
        info.is_mmproj = path.name.lower().startswith("mmproj")
        kv = _read_gguf_kv(path)
        arch = str(kv.get("general.architecture", "") or "")
        info.architecture = arch
        info.name = str(kv.get("general.name", "") or "")
        info.size_label = str(kv.get("general.size_label", "") or "")
        info.file_type = kv.get("general.file_type")
        p = arch or ""
        info.context_length = _as_int(kv.get(f"{p}.context_length") or kv.get("llama.context_length"))
        info.original_context_length = _as_int(kv.get(f"{p}.rope.scaling.original_context_length"))
        info.embedding_length = _as_int(kv.get(f"{p}.embedding_length"))
        info.block_count = _as_int(kv.get(f"{p}.block_count"))
        info.rope_freq_base = _as_float(kv.get(f"{p}.rope.freq_base"))
        info.rope_scaling_type = str(kv.get(f"{p}.rope.scaling.type", "") or "")
        info.rope_scaling_factor = _as_float(kv.get(f"{p}.rope.scaling.factor"))
        info.expert_count = _as_int(kv.get(f"{p}.expert_count"))
        info.expert_used_count = _as_int(kv.get(f"{p}.expert_used_count"))
    except Exception as exc:  # noqa: BLE001 - surface parse errors in UI
        info.error = str(exc)
    return info


def _read_gguf_kv(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = struct.unpack("<I", f.read(4))[0]
        if version < 2:
            raise ValueError(f"unsupported GGUF version: {version}")
        _n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        def read_string() -> str:
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", errors="replace")

        def skip(vtype: int) -> None:
            if vtype == 8:
                read_string()
            elif vtype == 9:
                (atype,) = struct.unpack("<I", f.read(4))
                (n,) = struct.unpack("<Q", f.read(8))
                if atype == 8:
                    for _ in range(n):
                        read_string()
                else:
                    f.read(n * GGUF_TYPE[atype][1])
            else:
                f.read(GGUF_TYPE[vtype][1])

        for _ in range(n_kv):
            key = read_string()
            (vtype,) = struct.unpack("<I", f.read(4))
            if vtype == 9:  # arrays (vocab) are large; skip
                skip(9)
                continue
            if vtype == 8:
                out[key] = read_string()
            else:
                out[key] = struct.unpack("<" + GGUF_TYPE[vtype][0], f.read(GGUF_TYPE[vtype][1]))[0]
    return out


# --------------------------------------------------------------------------
# model discovery
# --------------------------------------------------------------------------

def discover_models(models_dir: Path) -> list[ModelRef]:
    models_dir = Path(models_dir).expanduser()
    if not models_dir.is_dir():
        return []
    mmproj_by_dir: dict[Path, Path] = {}
    mains: list[Path] = []
    for p in sorted(models_dir.rglob("*.gguf")):
        if p.name.lower().startswith("mmproj"):
            mmproj_by_dir[p.parent] = p
        else:
            mains.append(p)
    return [ModelRef(path=p, mmproj=mmproj_by_dir.get(p.parent)) for p in mains]


def human_bytes(n: int) -> str:
    val = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if val < 1024.0 or unit == "TiB":
            return f"{int(val)} {unit}" if unit == "B" else f"{val:.2f} {unit}"
        val /= 1024.0
    return f"{n} B"


def estimate_vram_mib(size_bytes: int, ctx: int | None, block_count: int | None) -> float:
    """Very rough weights + KV estimate. Guidance only, not a calculator."""
    weights = size_bytes / (1024.0 * 1024.0) * 1.06
    ctx_n = ctx or 8192
    layers = block_count or 32
    kv = (ctx_n / 1024.0) * layers * 0.0625  # ~64 KiB per 1k ctx per layer
    return weights + kv


# --------------------------------------------------------------------------
# command building
# --------------------------------------------------------------------------

@dataclass
class LaunchOptions:
    mode: str
    model: ModelRef
    bin_dir: Path
    server_bin: str = "llama-server"
    cli_bin: str = "llama-completion"
    host: str = "127.0.0.1"
    port: int = 9898
    ngl: int = 99
    ctx: int = 0
    alias: str = ""
    api_key: str = ""
    use_mmproj: bool = True
    extra_args: list[str] = field(default_factory=list)


def resolve_bin(bin_dir: Path, name: str) -> Path:
    p = Path(name)
    if p.is_absolute():
        return p
    cand = Path(bin_dir).expanduser() / name
    if cand.exists():
        return cand
    which = shutil.which(name)
    return Path(which) if which else cand


def build_command(opts: LaunchOptions) -> list[str]:
    model_path = str(opts.model.path)
    mmproj = opts.model.mmproj
    want_mmproj = bool(opts.use_mmproj and mmproj and mmproj.exists())

    if opts.mode == "web":
        cmd = [
            str(resolve_bin(opts.bin_dir, opts.server_bin)),
            "-m", model_path,
            "-ngl", str(opts.ngl),
            "--host", opts.host,
            "--port", str(opts.port),
        ]
        if opts.ctx and opts.ctx > 0:
            cmd += ["-c", str(opts.ctx)]
        cmd += ["-a", opts.alias.strip() or opts.model.path.stem]
        if opts.api_key:
            cmd += ["--api-key", opts.api_key]
        if want_mmproj:
            cmd += ["--mmproj", str(mmproj)]
        return cmd + list(opts.extra_args)

    cli = resolve_bin(opts.bin_dir, opts.cli_bin)
    cmd = [str(cli), "-m", model_path, "-ngl", str(opts.ngl)]
    if opts.ctx and opts.ctx > 0:
        cmd += ["-c", str(opts.ctx)]
    if want_mmproj:
        cmd += ["-mm", str(mmproj)]
    if "--jinja" not in opts.extra_args and "--no-jinja" not in opts.extra_args:
        cmd.append("--jinja")
    return cmd + list(opts.extra_args)


def command_to_shell(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def find_cli_bin(bin_dir: Path, mmproj: bool, preferred: str = "llama-completion") -> Path | None:
    names = (["llama-cli", "llama-mtmd-cli", preferred] if mmproj else [preferred, "llama-cli"])
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        cand = resolve_bin(bin_dir, name)
        if cand.exists():
            return cand
    return None


# --------------------------------------------------------------------------
# server utilities
# --------------------------------------------------------------------------

def port_in_use(host: str, port: int) -> bool:
    check = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((check, port)) == 0


def server_url(opts: LaunchOptions) -> str:
    host = "127.0.0.1" if opts.host in ("0.0.0.0", "::", "") else opts.host
    return f"http://{host}:{opts.port}"


def probe_health(url: str, timeout: float = 1.0) -> tuple[bool, str]:
    """Return (reachable, status). status is 'ok' | 'loading' | 'error' | 'down'."""
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout) as resp:
            body = resp.read(4096).decode("utf-8", errors="replace")
            try:
                data = json.loads(body)
                status = str(data.get("status", "")).lower()
                if status in ("ok", "ready"):
                    return True, "ok"
                if status == "loading":
                    return True, "loading"
                if status:
                    return True, status
            except json.JSONDecodeError:
                pass
            return True, "ok"
    except urllib.error.HTTPError as exc:
        return True, "loading" if exc.code == 503 else "error"
    except (urllib.error.URLError, TimeoutError, OSError):
        return False, "down"


def open_in_browser(url: str) -> bool:
    for opener in ("xdg-open", "gio"):
        path = shutil.which(opener)
        if not path:
            continue
        try:
            args = [path, "open", url] if opener == "gio" else [path, url]
            subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except OSError:
            continue
    return False


def find_terminal() -> tuple[str, list[str]] | None:
    """Return (name, base_args) for launching an interactive terminal."""
    candidates = [
        ("gnome-terminal", ["gnome-terminal", "--"]),
        ("kgx", ["kgx", "--"]),
        ("konsole", ["konsole", "-e"]),
        ("xfce4-terminal", ["xfce4-terminal", "-x"]),
        ("alacritty", ["alacritty", "-e"]),
        ("kitty", ["kitty"]),
        ("x-terminal-emulator", ["x-terminal-emulator", "-e"]),
        ("xterm", ["xterm", "-e"]),
    ]
    for name, args in candidates:
        if shutil.which(name):
            return name, args
    return None


# --------------------------------------------------------------------------
# async process streaming
# --------------------------------------------------------------------------

@dataclass
class SpawnedJob:
    popen: subprocess.Popen
    mode: str
    label: str
    stop_event: threading.Event = field(default_factory=threading.Event)

    @property
    def pid(self) -> int:
        return self.popen.pid

    @property
    def running(self) -> bool:
        return self.popen.poll() is None

    def stop(self) -> None:
        self.stop_event.set()
        if self.popen.poll() is not None:
            return
        try:
            os.killpg(self.popen.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                self.popen.terminate()
            except OSError:
                return
        try:
            self.popen.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.popen.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    self.popen.kill()
                except OSError:
                    pass


def spawn_job(
    cmd: list[str],
    *,
    mode: str,
    label: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    on_line: Callable[[str], None] | None = None,
    on_done: Callable[[int], None] | None = None,
    idle_scheduler: Callable[[Callable[[], None]], None] | None = None,
) -> SpawnedJob:
    """Start cmd, stream merged stdout/stderr to on_line, call on_done(returncode).

    on_line/on_done are dispatched through idle_scheduler when given, so GTK
    widgets are only touched from the main thread.
    """
    dispatch = idle_scheduler or (lambda fn: fn())

    run_env = dict(os.environ)
    if env:
        run_env.update(env)

    popen = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        bufsize=1,
        text=True,
        errors="replace",
    )
    job = SpawnedJob(popen=popen, mode=mode, label=label)

    def pump_stdout() -> None:
        assert popen.stdout is not None
        for raw in iter(popen.stdout.readline, ""):
            if on_line:
                dispatch(lambda line=raw.rstrip("\n"): on_line(line))
        popen.stdout.close()
        code = popen.wait()
        if on_done:
            dispatch(lambda c=code: on_done(c))

    threading.Thread(target=pump_stdout, name=f"stream-{mode}", daemon=True).start()
    return job


def build_script_cmd(*, clean: bool = False, upstream_branch: str = "", target_branch: str = "",
                     cuda_arch: str = "", jobs: int = 0, build_dir: str = "") -> tuple[list[str], dict[str, str]]:
    cmd = ["bash", str(BUILD_SCRIPT)]
    if clean:
        cmd.append("--clean")
    env: dict[str, str] = {}
    if upstream_branch:
        env["UPSTREAM_BRANCH"] = upstream_branch
    if target_branch:
        env["TARGET_BRANCH"] = target_branch
    if cuda_arch:
        env["CUDA_ARCH"] = cuda_arch
    if jobs and jobs > 0:
        env["JOBS"] = str(jobs)
    if build_dir:
        env["BUILD_DIR"] = build_dir
    return cmd, env


def git_info(cwd: Path | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for key, args in (
            ("branch", ["rev-parse", "--abbrev-ref", "HEAD"]),
            ("head", ["rev-parse", "--short", "HEAD"]),
            ("upstream", ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]),
            ("dirty", ["status", "--porcelain"]),
        ):
            proc = subprocess.run(["git", *args], cwd=str(cwd or REPO_ROOT),
                                  capture_output=True, text=True, timeout=5)
            if proc.returncode == 0:
                out[key] = proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return out


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
