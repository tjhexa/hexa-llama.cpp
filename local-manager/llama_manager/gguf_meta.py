"""Minimal GGUF metadata reader (stdlib only, no numpy)."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


GGUF_TYPE = {
    0: ("UINT8", "B", 1),
    1: ("INT8", "b", 1),
    2: ("UINT16", "H", 2),
    3: ("INT16", "h", 2),
    4: ("UINT32", "I", 4),
    5: ("INT32", "i", 4),
    6: ("FLOAT32", "f", 4),
    7: ("BOOL", "?", 1),
    8: ("STRING", None, None),
    9: ("ARRAY", None, None),
    10: ("UINT64", "Q", 8),
    11: ("INT64", "q", 8),
    12: ("FLOAT64", "d", 8),
}


@dataclass
class GgufInfo:
    path: Path
    size_bytes: int = 0
    architecture: str = ""
    name: str = ""
    basename: str = ""
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
    sliding_window: int | None = None
    is_mmproj: bool = False
    kv: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def size_human(self) -> str:
        return _human_bytes(self.size_bytes)

    @property
    def display_name(self) -> str:
        return self.name or self.basename or self.path.stem

    @property
    def quant_hint(self) -> str:
        # Prefer filename quant tags, then GGUF file_type if present.
        name = self.path.name.upper()
        for tag in (
            "MXFP4",
            "IQ4_NL",
            "IQ4_XS",
            "IQ3_XXS",
            "IQ3_M",
            "IQ2_XXS",
            "IQ2_M",
            "IQ1_M",
            "IQ1_S",
            "Q8_0",
            "Q6_K",
            "Q5_K_M",
            "Q5_K_S",
            "Q5_0",
            "Q4_K_M",
            "Q4_K_S",
            "Q4_0",
            "Q3_K_M",
            "Q3_K_S",
            "Q2_K",
            "Q1_0",
            "BF16",
            "F16",
            "F32",
        ):
            if tag in name:
                return tag
        if self.file_type is not None:
            return f"type={self.file_type}"
        return "?"


def _human_bytes(n: int) -> str:
    val = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if val < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(val)} {unit}"
            return f"{val:.2f} {unit}"
        val /= 1024.0
    return f"{n} B"


def read_gguf_info(path: Path) -> GgufInfo:
    info = GgufInfo(path=path)
    try:
        info.size_bytes = path.stat().st_size
        info.is_mmproj = path.name.lower().startswith("mmproj")
        kv = _read_gguf_kv(path)
        info.kv = kv
        arch = str(kv.get("general.architecture", "") or "")
        info.architecture = arch
        info.name = str(kv.get("general.name", "") or "")
        info.basename = str(kv.get("general.basename", "") or "")
        info.size_label = str(kv.get("general.size_label", "") or "")
        info.file_type = kv.get("general.file_type")

        prefix = arch if arch else ""
        info.context_length = _as_int(
            kv.get(f"{prefix}.context_length")
            or kv.get("llama.context_length")
            or kv.get("general.context_length")
        )
        info.original_context_length = _as_int(
            kv.get(f"{prefix}.rope.scaling.original_context_length")
        )
        info.embedding_length = _as_int(kv.get(f"{prefix}.embedding_length"))
        info.block_count = _as_int(kv.get(f"{prefix}.block_count"))
        info.rope_freq_base = _as_float(kv.get(f"{prefix}.rope.freq_base"))
        info.rope_scaling_type = str(
            kv.get(f"{prefix}.rope.scaling.type", "") or ""
        )
        info.rope_scaling_factor = _as_float(
            kv.get(f"{prefix}.rope.scaling.factor")
        )
        info.expert_count = _as_int(kv.get(f"{prefix}.expert_count"))
        info.expert_used_count = _as_int(kv.get(f"{prefix}.expert_used_count"))
        info.sliding_window = _as_int(
            kv.get(f"{prefix}.attention.sliding_window")
        )
    except Exception as exc:  # noqa: BLE001 - surface parse errors in UI
        info.error = str(exc)
    return info


def _as_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _read_gguf_kv(path: Path) -> dict[str, Any]:
    """Read scalar/string KV metadata; skip large arrays by counting only."""
    out: dict[str, Any] = {}
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError("not a GGUF file")
        version = struct.unpack("<I", f.read(4))[0]
        if version < 2:
            raise ValueError(f"unsupported GGUF version: {version}")
        _n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        def read_string() -> str:
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", errors="replace")

        def skip_value(vtype: int) -> None:
            if vtype == 8:
                read_string()
                return
            if vtype == 9:
                (atype,) = struct.unpack("<I", f.read(4))
                (n,) = struct.unpack("<Q", f.read(8))
                if atype == 8:
                    for _ in range(n):
                        read_string()
                    return
                _name, _fmt, size = GGUF_TYPE[atype]
                f.read(n * size)
                return
            _name, _fmt, size = GGUF_TYPE[vtype]
            f.read(size)

        def read_value(vtype: int) -> Any:
            if vtype == 8:
                return read_string()
            if vtype == 9:
                # Arrays (tokens/merges) can be huge; skip payload.
                skip_value(9)
                return "<array>"
            name, fmt, size = GGUF_TYPE[vtype]
            (val,) = struct.unpack("<" + fmt, f.read(size))
            return val

        for _ in range(n_kv):
            key = read_string()
            (vtype,) = struct.unpack("<I", f.read(4))
            # Keep scalars/strings; skip arrays for speed.
            if vtype == 9:
                skip_value(9)
                continue
            out[key] = read_value(vtype)
    return out
