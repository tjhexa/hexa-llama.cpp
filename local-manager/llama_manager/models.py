"""Model discovery under models_dir."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .gguf_meta import GgufInfo, read_gguf_info


@dataclass
class ModelEntry:
    info: GgufInfo
    mmproj: Path | None = None

    @property
    def path(self) -> Path:
        return self.info.path


def discover_models(
    models_dir: Path,
    *,
    include_mmproj: bool = False,
) -> list[ModelEntry]:
    if not models_dir.is_dir():
        return []

    ggufs = sorted(models_dir.rglob("*.gguf"))
    mmproj_map: dict[Path, Path] = {}
    mains: list[Path] = []

    for p in ggufs:
        if p.name.lower().startswith("mmproj"):
            mmproj_map[p.parent] = p
        else:
            mains.append(p)

    entries: list[ModelEntry] = []
    for p in mains:
        info = read_gguf_info(p)
        entries.append(ModelEntry(info=info, mmproj=mmproj_map.get(p.parent)))

    if include_mmproj:
        for parent, mp in sorted(mmproj_map.items(), key=lambda x: str(x[1])):
            # Only add orphan mmproj if somehow not paired; usually skip.
            if all(e.mmproj != mp for e in entries):
                entries.append(ModelEntry(info=read_gguf_info(mp), mmproj=None))

    return entries


def find_by_path(entries: list[ModelEntry], path: str | Path) -> ModelEntry | None:
    target = Path(path).resolve()
    for e in entries:
        if e.path.resolve() == target:
            return e
    # Also allow matching by filename stem
    name = Path(path).name
    for e in entries:
        if e.path.name == name:
            return e
    return None
