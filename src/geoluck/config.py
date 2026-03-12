from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_raw: Path
    data_intermediate: Path
    data_final: Path
    data_web: Path
    docs: Path
    do: Path
    web: Path
    web_public: Path


def project_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate project root from pyproject.toml")


def get_paths(start: Path | None = None) -> ProjectPaths:
    root = project_root(start)
    return ProjectPaths(
        root=root,
        data_raw=root / "data_raw",
        data_intermediate=root / "data_intermediate",
        data_final=root / "data_final",
        data_web=root / "data_final" / "web",
        docs=root / "docs",
        do=root / "do",
        web=root / "web",
        web_public=root / "web" / "public",
    )
