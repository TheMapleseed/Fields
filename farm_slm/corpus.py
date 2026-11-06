"""Normalize corpus JSONL into training examples the model can consume."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from farm_slm.locality import LocalityError, condition_text, flatten_text, hold_out_region, normalize_region

# Catalog and map labels folded into the five model domains.
_DOMAIN_ALIASES: dict[str, str] = {
    "english": "english",
    "agricultural_linguistics": "agricultural_linguistics",
    "chemistry": "chemistry",
    "weather_reading": "weather_reading",
    "physics": "physics",
    "spatial": "physics",
    "soil": "chemistry",
    "agriculture": "agricultural_linguistics",
    "botany": "agricultural_linguistics",
    "mathematics": "physics",
    "topology": "physics",
    "biology": "chemistry",
    # OpenStax and Mises economics fold into farm-facing market language.
    "economics": "agricultural_linguistics",
}


class CorpusError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingExample:
    domain: str
    text: str
    region: str
    units: str
    chip: Path | None = None


def resolve_domain(domain: str, allowed: Sequence[str]) -> str:
    key = domain.strip().lower()
    mapped = _DOMAIN_ALIASES.get(key, key)
    if mapped not in allowed:
        raise CorpusError(f"domain {domain!r} does not map to a model domain")
    return mapped


def example_from_record(
    row: Mapping[str, object],
    allowed_domains: Sequence[str],
    *,
    chip_root: Path | None = None,
) -> TrainingExample | None:
    if not isinstance(row, dict):
        return None
    raw_domain = row.get("domain")
    if not isinstance(raw_domain, str) or not raw_domain.strip():
        return None
    try:
        domain = resolve_domain(raw_domain, allowed_domains)
    except CorpusError:
        return None
    text = _record_text(row)
    if text is None:
        return None
    region = row.get("region", "unspecified")
    units = row.get("units", "unspecified")
    if not isinstance(region, str) or not isinstance(units, str):
        return None
    try:
        conditioned = condition_text(flatten_text(text), region, units)
        place = normalize_region(region)
    except LocalityError:
        return None
    chip = _chip_path(row, chip_root)
    unit_value = units.strip().lower() if units.strip() else "unspecified"
    return TrainingExample(
        domain=domain,
        text=conditioned,
        region=place,
        units=unit_value if unit_value else "unspecified",
        chip=chip,
    )


def load_training_examples(
    root: Path,
    domains: Sequence[str],
    *,
    hold_out: str = "none",
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    corpus = root / "data" / "corpus"
    maps = root / "data" / "maps"
    rows: list[dict[str, object]] = []
    if corpus.is_dir():
        rows.extend(_read_jsonl_dir(corpus))
    if maps.is_dir():
        rows.extend(_read_jsonl_dir(maps))
    examples: list[TrainingExample] = []
    for row in rows:
        example = example_from_record(row, domains, chip_root=maps if maps.is_dir() else None)
        if example is not None:
            examples.append(example)
    if hold_out.strip().lower() in {"", "none", "unspecified"}:
        return examples, []
    as_maps = [
        {
            "domain": item.domain,
            "text": item.text,
            "region": item.region,
            "units": item.units,
            "chip": str(item.chip) if item.chip is not None else "",
        }
        for item in examples
    ]
    try:
        train_rows, held_rows = hold_out_region(as_maps, hold_out)
    except LocalityError as exc:
        raise CorpusError(str(exc)) from exc
    return _examples_from_plain(train_rows), _examples_from_plain(held_rows)


def _examples_from_plain(rows: Sequence[Mapping[str, str]]) -> list[TrainingExample]:
    out: list[TrainingExample] = []
    for row in rows:
        chip_value = row.get("chip", "")
        chip = Path(chip_value) if chip_value else None
        out.append(
            TrainingExample(
                domain=row["domain"],
                text=row["text"],
                region=row["region"],
                units=row["units"],
                chip=chip if chip is not None and chip_value else None,
            )
        )
    return out


def _record_text(row: Mapping[str, object]) -> str | None:
    text = row.get("text")
    if isinstance(text, str) and text.strip():
        return text
    instruction = row.get("instruction")
    prompt = row.get("input")
    output = row.get("output")
    if (
        isinstance(instruction, str)
        and isinstance(prompt, str)
        and isinstance(output, str)
        and instruction.strip()
        and prompt.strip()
        and output.strip()
    ):
        return f"Instruction: {instruction.strip()} Input: {prompt.strip()} Output: {output.strip()}"
    return None


def _chip_path(row: Mapping[str, object], chip_root: Path | None) -> Path | None:
    chip = row.get("chip")
    if not isinstance(chip, str) or not chip.strip():
        return None
    candidate = Path(chip.strip())
    if candidate.is_file():
        return candidate.resolve()
    if chip_root is None:
        return None
    name = candidate.name
    for path in (chip_root / candidate, chip_root / "chips" / name, chip_root / name):
        if path.is_file():
            return path.resolve()
    for path in chip_root.rglob(name):
        if path.is_file():
            return path.resolve()
    return None


def _read_jsonl_dir(directory: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.jsonl")):
        if path.name.endswith(".partial"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows
