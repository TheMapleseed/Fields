"""Deferred dump and binary-document intake.

Catalog items on deferred_dump and deferred_document channels are listed here.
Readers are not enabled yet; this module does not download or unpack archives.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from farm_slm.config import project_root
from farm_slm.sources import DeferredTarget, SourceCatalog, dump_targets_from_catalog


class DumpError(RuntimeError):
    pass


@dataclass(frozen=True)
class DumpPlan:
    targets: tuple[DeferredTarget, ...]
    staging_dir: Path


def dump_staging_directory(root: Path | None = None) -> Path:
    base = project_root() if root is None else root
    return base / "data" / "dumps"


def catalog_dump_plan(catalog: SourceCatalog | None = None, root: Path | None = None) -> DumpPlan:
    return DumpPlan(
        targets=dump_targets_from_catalog(catalog),
        staging_dir=dump_staging_directory(root),
    )


def dump_intake_report(catalog: SourceCatalog | None = None, root: Path | None = None) -> str:
    plan = catalog_dump_plan(catalog, root)
    staging = plan.staging_dir
    local = sorted(staging.iterdir()) if staging.is_dir() else []
    lines = [
        "Dump and document intake",
        f"catalog_targets={len(plan.targets)}",
        f"staging={staging.as_posix()}",
        f"local_files={len(local)}",
    ]
    for target in plan.targets:
        lines.append(f"  - {target.channel}: {target.title} [{target.license}] {target.url}")
    if local:
        for path in local[:5]:
            lines.append(f"  file: {path.name}")
        if len(local) > 5:
            lines.append(f"  file: … {len(local) - 5} more")
    else:
        lines.append("  (no staged dumps yet)")
    lines.append("Readers are not enabled; place archives under data/dumps/ when ready.")
    return "\n".join(lines)


def read_dump_text(path: Path) -> str:
    """Reserved for dump readers. Refuses until a format-specific reader exists."""
    raise DumpError(
        f"no dump reader is registered for {path.name}; catalog dump targets stay deferred"
    )
