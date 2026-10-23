"""Place and unit system for a training record.

The tokenizer is bytes, so a local language needs no new token. The model
learns a place only when the text names it. Region tokens are not added to the
vocabulary. A record with no known place uses region "unspecified".
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from farm_slm.erosion import (
    erodibility_to_us_customary,
    optimize_erosion,
    rainfall_to_us_customary,
    tons_per_acre_to_reported,
)

_REGION = re.compile(r"^[a-z]{2}$|^[a-z]+(?:-[a-z]+)*$")
_UNITS = {"us-customary", "si", "unspecified"}
_UNIT_LABEL = {"us-customary": "US customary", "si": "SI"}


class LocalityError(ValueError):
    pass


def normalize_region(region: str) -> str:
    cleaned = region.strip().lower()
    if not cleaned or len(cleaned) > 40 or _REGION.fullmatch(cleaned) is None:
        raise LocalityError("region must be a lowercase place name or an ISO country code")
    return cleaned


def normalize_units(units: str) -> str:
    cleaned = units.strip().lower()
    if cleaned not in _UNITS:
        raise LocalityError("units must be us-customary, si, or unspecified")
    return cleaned


def condition_text(text: str, region: str, units: str) -> str:
    """Put the place and the unit system in front of the text when they are known."""
    place = normalize_region(region)
    system = normalize_units(units)
    body = text.strip()
    if not body or "\n" in body or "\r" in body:
        raise LocalityError("training text must be one non-empty line")
    parts: list[str] = []
    if place != "unspecified":
        parts.append(f"Place: {place}.")
    if system != "unspecified":
        parts.append(f"Units: {_UNIT_LABEL[system]}.")
    if not parts:
        return body
    prefix = " ".join(parts)
    if body.startswith(prefix):
        return body
    return f"{prefix} {body}"


def hold_out_region(
    records: Sequence[Mapping[str, str]],
    region: str,
) -> tuple[tuple[Mapping[str, str], ...], tuple[Mapping[str, str], ...]]:
    """Keep one place out of training so a score is not only that place memorized."""
    held_region = normalize_region(region)
    if held_region == "unspecified":
        raise LocalityError("a held-out place has to be named")
    train: list[Mapping[str, str]] = []
    held: list[Mapping[str, str]] = []
    for record in records:
        record_region = normalize_region(str(record.get("region", "")))
        if record_region == held_region:
            held.append(record)
        else:
            train.append(record)
    if not held:
        raise LocalityError(f"no records for {held_region}")
    return tuple(train), tuple(held)


def erosion_training_record(
    *,
    domain: str,
    region: str,
    units: str,
    rainfall_erosivity: float,
    soil_erodibility: float,
    slope_length_feet: float,
    slope_percent: float,
    license_id: str,
    source_url: str,
) -> dict[str, str]:
    """Describe one slope in the caller's units, after calculating loss in US customary units."""
    place = normalize_region(region)
    system = normalize_units(units)
    if place == "unspecified" or system == "unspecified":
        raise LocalityError("an erosion example needs a place and a unit system")
    if not license_id.strip() or not source_url.startswith("https://"):
        raise LocalityError("an erosion example needs a license and an https source")
    rainfall = rainfall_to_us_customary(rainfall_erosivity, system)
    erodibility = erodibility_to_us_customary(soil_erodibility, system)
    result = optimize_erosion(rainfall, erodibility, slope_length_feet, slope_percent)
    chosen = result.chosen
    reported = tons_per_acre_to_reported(chosen.tons_per_acre_year, system)
    tolerance = tons_per_acre_to_reported(chosen.tolerance_tons_per_acre_year, system)
    if system == "si":
        loss_words = f"{reported:.4f} tonnes per hectare per year"
        tolerance_words = f"{tolerance:.4f} tonnes per hectare per year"
        length_words = f"{slope_length_feet * 0.3048:.2f} meters"
    else:
        loss_words = f"{reported:.4f} tons per acre per year"
        tolerance_words = f"{tolerance:.4f} tons per acre per year"
        length_words = f"{slope_length_feet:.1f} feet"
    status = "meets" if chosen.meets_tolerance else "exceeds"
    prose = (
        f"Rainfall erosivity R is {rainfall_erosivity}. "
        f"Soil erodibility K is {soil_erodibility}. "
        f"Slope length is {length_words}. Slope is {slope_percent} percent. "
        f"The lowest sheet and rill loss is {loss_words} "
        f"with cover {chosen.cover} and support {chosen.support}. "
        f"The planning tolerance is {tolerance_words}. This plan {status} that tolerance."
    )
    return {
        "domain": domain.strip(),
        "region": place,
        "units": system,
        "license": license_id.strip(),
        "source_url": source_url,
        "text": condition_text(prose, place, system),
    }
