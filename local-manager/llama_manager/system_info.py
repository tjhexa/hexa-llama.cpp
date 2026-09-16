"""System RAM / VRAM helpers."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class GpuInfo:
    index: int
    name: str
    total_mib: int
    used_mib: int
    free_mib: int

    @property
    def total_gib(self) -> float:
        return self.total_mib / 1024.0

    @property
    def used_gib(self) -> float:
        return self.used_mib / 1024.0

    @property
    def free_gib(self) -> float:
        return self.free_mib / 1024.0


@dataclass
class SystemInfo:
    ram_total_kib: int = 0
    ram_available_kib: int = 0
    ram_used_kib: int = 0
    gpus: list[GpuInfo] = field(default_factory=list)
    nvidia_error: str = ""

    @property
    def ram_total_gib(self) -> float:
        return self.ram_total_kib / (1024.0 * 1024.0)

    @property
    def ram_available_gib(self) -> float:
        return self.ram_available_kib / (1024.0 * 1024.0)

    @property
    def ram_used_gib(self) -> float:
        return self.ram_used_kib / (1024.0 * 1024.0)

    @property
    def vram_total_mib(self) -> int:
        return sum(g.total_mib for g in self.gpus)

    @property
    def vram_free_mib(self) -> int:
        return sum(g.free_mib for g in self.gpus)


def collect_system_info() -> SystemInfo:
    info = SystemInfo()
    _fill_ram(info)
    _fill_gpus(info)
    return info


def _fill_ram(info: SystemInfo) -> None:
    try:
        mem: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                num = parts[1].strip().split()[0]
                mem[key] = int(num)
        info.ram_total_kib = mem.get("MemTotal", 0)
        info.ram_available_kib = mem.get("MemAvailable", 0)
        info.ram_used_kib = max(
            0, info.ram_total_kib - info.ram_available_kib
        )
    except OSError:
        pass


def _fill_gpus(info: SystemInfo) -> None:
    if not shutil.which("nvidia-smi"):
        info.nvidia_error = "nvidia-smi not found"
        return
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        info.nvidia_error = str(exc)
        return

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        info.nvidia_error = err or f"nvidia-smi exit {proc.returncode}"
        return

    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            info.gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    total_mib=int(float(parts[2])),
                    used_mib=int(float(parts[3])),
                    free_mib=int(float(parts[4])),
                )
            )
        except ValueError:
            continue


def format_system_banner(info: SystemInfo) -> str:
    lines = [
        "System resources",
        f"  RAM  total={info.ram_total_gib:.1f} GiB"
        f"  used={info.ram_used_gib:.1f} GiB"
        f"  available={info.ram_available_gib:.1f} GiB",
    ]
    if info.gpus:
        for g in info.gpus:
            lines.append(
                f"  GPU{g.index} {g.name}"
                f"  VRAM total={g.total_gib:.1f} GiB"
                f"  used={g.used_gib:.1f} GiB"
                f"  free={g.free_gib:.1f} GiB"
            )
    else:
        lines.append(f"  GPU  unavailable ({info.nvidia_error or 'none'})")
    return "\n".join(lines)


def rough_fit_hint(
    model_size_bytes: int,
    ctx: int | None,
    vram_free_mib: int,
) -> str:
    """Very rough guidance only; not a precise calculator."""
    if vram_free_mib <= 0:
        return "No VRAM reading; use -ngl carefully or run CPU-only (-ngl 0)."
    model_mib = model_size_bytes / (1024.0 * 1024.0)
    # Assume weights mostly on GPU + some KV overhead.
    ctx_n = ctx or 8192
    # Extremely rough KV estimate: ~0.5 MiB per 1k ctx for mid models.
    kv_mib = (ctx_n / 1000.0) * 0.5 * 24
    need = model_mib * 1.05 + kv_mib
    free = float(vram_free_mib)
    if need < free * 0.85:
        return (
            f"Rough fit OK for ctx~{ctx_n}"
            f" (est ~{need:.0f} MiB vs free {free:.0f} MiB)."
        )
    if model_mib < free:
        return (
            f"Weights may fit, but ctx~{ctx_n} may be tight"
            f" (est ~{need:.0f} MiB vs free {free:.0f} MiB)."
            " Lower -c or use -ctk q8_0."
        )
    return (
        f"Likely too large for full GPU offload"
        f" (model ~{model_mib:.0f} MiB, free VRAM {free:.0f} MiB)."
        " Use lower -ngl, CPU MoE offload, or a smaller quant."
    )


def port_in_use(host: str, port: int) -> bool:
    import socket

    check_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((check_host, port)) == 0

