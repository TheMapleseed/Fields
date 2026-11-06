"""Persist ephemeral training stage next to model checkpoints.

The stage records which catalog/PDF job and page to resume after a stop.
It is written atomically as stage.kdl beside the safetensors checkpoints.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from farm_slm.ephemeral import EphemeralJob
from farm_slm.kdl import KdlError, parse_kdl

_STAGE_NAME = "stage.kdl"
_FORMAT = "farm_slm_stage"


class StageError(ValueError):
    pass


@dataclass(frozen=True)
class StageState:
    step: int
    job_index: int
    page_index: int
    job_url: str
    job_title: str
    jobs_sha: str
    order: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.step < 0 or self.job_index < 0 or self.page_index < 0:
            raise StageError("stage indexes cannot be negative")
        if "\n" in self.job_url or "\r" in self.job_url:
            raise StageError("job_url cannot contain newlines")
        if "\n" in self.job_title or "\r" in self.job_title:
            raise StageError("job_title cannot contain newlines")
        if len(self.jobs_sha) > 128 or any(char.isspace() for char in self.jobs_sha):
            raise StageError("jobs_sha is malformed")


def stage_path(directory: Path) -> Path:
    return directory / _STAGE_NAME


def stage_exists(directory: Path) -> bool:
    return stage_path(directory).is_file()


def jobs_fingerprint(jobs: tuple[EphemeralJob, ...] | list[EphemeralJob]) -> str:
    digest = hashlib.sha256()
    for job in jobs:
        digest.update(
            f"{job.channel}\0{job.url}\0{job.title}\0{job.model_domain}\n".encode("utf-8")
        )
    return digest.hexdigest()


def save_stage(directory: Path, state: StageState) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = stage_path(directory)
    temporary = destination.with_suffix(".kdl.partial")
    order = ",".join(str(value) for value in state.order)
    text = (
        "stage {\n"
        f'  format "{_FORMAT}"\n'
        f"  step {state.step}\n"
        f"  job_index {state.job_index}\n"
        f"  page_index {state.page_index}\n"
        f"  job_url {_quote(state.job_url)}\n"
        f"  job_title {_quote(state.job_title)}\n"
        f'  jobs_sha "{state.jobs_sha}"\n'
        f'  order "{order}"\n'
        "}\n"
    )
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_stage(directory: Path) -> StageState:
    path = stage_path(directory)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StageError(f"cannot read {path}") from exc
    try:
        nodes = parse_kdl(text)
    except KdlError as exc:
        raise StageError(f"invalid stage file: {exc}") from exc
    matches = [node for node in nodes if node.name == "stage"]
    if len(matches) != 1:
        raise StageError("stage.kdl must contain one stage node")
    node = matches[0]
    if _string(node, "format") != _FORMAT:
        raise StageError("stage file is not a FarmSLM stage")
    order_raw = _string(node, "order")
    order: tuple[int, ...] = ()
    if order_raw.strip():
        try:
            order = tuple(int(part) for part in order_raw.split(",") if part != "")
        except ValueError as exc:
            raise StageError("stage order is malformed") from exc
    return StageState(
        step=_int(node, "step"),
        job_index=_int(node, "job_index"),
        page_index=_int(node, "page_index"),
        job_url=_string(node, "job_url"),
        job_title=_string(node, "job_title"),
        jobs_sha=_string(node, "jobs_sha"),
        order=order,
    )


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _string(node: object, name: str) -> str:
    from farm_slm.kdl import Node

    assert isinstance(node, Node)
    matches = [child for child in node.children if child.name == name]
    if len(matches) != 1 or len(matches[0].args) != 1 or not isinstance(matches[0].args[0], str):
        raise StageError(f"stage is missing string {name}")
    return matches[0].args[0]


def _int(node: object, name: str) -> int:
    from farm_slm.kdl import Node

    assert isinstance(node, Node)
    matches = [child for child in node.children if child.name == name]
    if len(matches) != 1 or len(matches[0].args) != 1:
        raise StageError(f"stage is missing integer {name}")
    value = matches[0].args[0]
    if isinstance(value, bool) or not isinstance(value, int):
        raise StageError(f"stage {name} must be an integer")
    return value
