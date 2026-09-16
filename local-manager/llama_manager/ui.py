"""Interactive terminal UI helpers."""

from __future__ import annotations

from pathlib import Path

from .models import ModelEntry
from .system_info import SystemInfo, format_system_banner, rough_fit_hint


def clear_line() -> None:
    print()


def print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_system(info: SystemInfo) -> None:
    print(format_system_banner(info))


def format_model_row(idx: int, entry: ModelEntry, last: str = "") -> str:
    info = entry.info
    mark = " *" if last and str(entry.path) == last else ""
    ctx = (
        f"{info.context_length}"
        if info.context_length is not None
        else "?"
    )
    arch = info.architecture or "?"
    quant = info.quant_hint
    size = info.size_human
    mm = " +mmproj" if entry.mmproj else ""
    name = info.display_name
    rel = _short_path(entry.path)
    return (
        f"  [{idx}]{mark} {rel}\n"
        f"       name={name}  arch={arch}  quant={quant}"
        f"  size={size}  ctx={ctx}{mm}"
    )


def print_model_list(
    entries: list[ModelEntry],
    *,
    last_model: str = "",
) -> None:
    if not entries:
        print("  (no .gguf models found)")
        return
    for i, e in enumerate(entries, start=1):
        print(format_model_row(i, e, last_model))
        if e.info.error:
            print(f"       WARN metadata: {e.info.error}")


def print_model_details(
    entry: ModelEntry,
    sysinfo: SystemInfo | None = None,
    ctx_for_hint: int | None = None,
) -> None:
    info = entry.info
    print_header(f"Model details: {entry.path.name}")
    print(f"  path         : {entry.path}")
    print(f"  display name : {info.display_name}")
    print(f"  architecture : {info.architecture or '?'}")
    print(f"  size label   : {info.size_label or '?'}")
    print(f"  file size    : {info.size_human}")
    print(f"  quant hint   : {info.quant_hint}")
    print(f"  max context  : {_fmt_ctx(info.context_length)}")
    if info.original_context_length:
        print(f"  orig context : {info.original_context_length}")
    if info.rope_scaling_type:
        print(
            f"  rope scaling : {info.rope_scaling_type}"
            f" factor={info.rope_scaling_factor}"
        )
    if info.embedding_length:
        print(f"  embedding    : {info.embedding_length}")
    if info.block_count:
        print(f"  layers       : {info.block_count}")
    if info.expert_count:
        print(
            f"  MoE experts  : {info.expert_count}"
            f" (used={info.expert_used_count})"
        )
    if info.sliding_window:
        print(f"  sliding win  : {info.sliding_window}")
    if entry.mmproj:
        print(f"  mmproj       : {entry.mmproj.name}")
    else:
        print("  mmproj       : (none found in folder)")
    if info.error:
        print(f"  metadata err : {info.error}")
    if sysinfo is not None:
        hint = rough_fit_hint(
            info.size_bytes,
            ctx_for_hint or info.context_length,
            sysinfo.vram_free_mib,
        )
        print(f"  fit hint     : {hint}")


def _fmt_ctx(n: int | None) -> str:
    if n is None:
        return "unknown"
    if n >= 1024:
        return f"{n} ({n / 1024:.0f}K)"
    return str(n)


def _short_path(path: Path) -> str:
    parts = path.parts
    if "models" in parts:
        i = parts.index("models")
        return str(Path(*parts[i:]))
    return path.name


def prompt_choice(prompt: str, valid: set[str], default: str = "") -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default:
            return default
        if raw in valid:
            return raw
        print(f"  Invalid choice. Expected one of: {', '.join(sorted(valid))}")


def prompt_int(
    prompt: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = int(raw)
        except ValueError:
            print("  Enter an integer.")
            continue
        if minimum is not None and val < minimum:
            print(f"  Must be >= {minimum}")
            continue
        if maximum is not None and val > maximum:
            print(f"  Must be <= {maximum}")
            continue
        return val


def prompt_yn(prompt: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{d}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true")


def prompt_str(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    return raw if raw else default


def select_model_index(
    entries: list[ModelEntry],
    *,
    last_model: str = "",
) -> int:
    if not entries:
        raise SystemExit("No models found. Put .gguf files in models_dir.")
    default = 1
    if last_model:
        for i, e in enumerate(entries, start=1):
            if str(e.path) == last_model:
                default = i
                break
    while True:
        raw = input(
            f"Select model number (d=details, q=quit) [{default}]: "
        ).strip().lower()
        if not raw:
            return default
        if raw in ("q", "quit", "exit"):
            raise SystemExit(0)
        if raw.startswith("d"):
            # d / d3
            part = raw[1:].strip()
            idx = default if not part else _parse_idx(part, len(entries))
            if idx is None:
                print("  Usage: d  or  d<number>  e.g. d2")
                continue
            print_model_details(entries[idx - 1])
            continue
        idx = _parse_idx(raw, len(entries))
        if idx is None:
            print(f"  Enter 1..{len(entries)}, d#, or q")
            continue
        return idx


def _parse_idx(raw: str, n: int) -> int | None:
    try:
        idx = int(raw)
    except ValueError:
        return None
    if 1 <= idx <= n:
        return idx
    return None
