"""Soil loss and the practices that keep it down.

Annual sheet and rill erosion follows the Revised Universal Soil Loss Equation
in US customary units, from USDA Agriculture Handbook 703:

    A = R * K * L * S * C * P

A is tons per acre per year. R is rainfall erosivity. K is soil erodibility.
L and S come from slope length in feet and slope in percent. C is the cover
factor and P is the support-practice factor. Both are greater than 0 and at
most 1.

The support factors are the contour and contour-strip tables in Wischmeier and
Smith, as republished in the Agriculture and Agri-Food Canada RUSLEFAC manual.
A practice is skipped when the slope class is outside its table or the slope
is longer than that table allows. The search keeps the cover and support pair
with the lowest soil loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from farm_slm.config import ErosionSettings, SupportPractice, load_erosion_settings

_REFERENCE_FEET = 72.6
_SHORT_SLOPE_FEET = 15.0
_FEET_TO_METERS = 0.3048
# Agriculture Handbook 703 conversions between US customary and SI.
_R_SI_PER_US = 17.02
_K_SI_PER_US = 0.1317
_TONNES_PER_HECTARE_PER_TON_PER_ACRE = 2.242


class ErosionError(ValueError):
    pass


@dataclass(frozen=True)
class SoilLoss:
    tons_per_acre_year: float
    length_factor: float
    steepness_factor: float
    cover_factor: float
    practice_factor: float


@dataclass(frozen=True)
class ErosionChoice:
    cover: str
    support: str
    tons_per_acre_year: float
    length_factor: float
    steepness_factor: float
    cover_factor: float
    practice_factor: float
    meets_tolerance: bool
    tolerance_tons_per_acre_year: float


@dataclass(frozen=True)
class ErosionOptimization:
    chosen: ErosionChoice
    ranked: tuple[ErosionChoice, ...]


def annual_soil_loss(
    rainfall_erosivity: float,
    soil_erodibility: float,
    slope_length_feet: float,
    slope_percent: float,
    cover_factor: float,
    practice_factor: float,
) -> SoilLoss:
    """Return RUSLE soil loss for one slope in tons per acre per year."""
    _positive_finite("rainfall erosivity", rainfall_erosivity)
    _positive_finite("soil erodibility", soil_erodibility)
    _positive_finite("slope length", slope_length_feet)
    _percent("slope", slope_percent)
    _factor("cover factor", cover_factor)
    _factor("practice factor", practice_factor)
    length = length_factor(slope_length_feet, slope_percent)
    steepness = steepness_factor(slope_length_feet, slope_percent)
    loss = (
        rainfall_erosivity
        * soil_erodibility
        * length
        * steepness
        * cover_factor
        * practice_factor
    )
    return SoilLoss(loss, length, steepness, cover_factor, practice_factor)


def length_factor(slope_length_feet: float, slope_percent: float) -> float:
    sin_theta = _sin_slope(slope_percent)
    beta = (sin_theta / 0.0896) / (3.0 * (sin_theta**0.8) + 0.56)
    exponent = beta / (1.0 + beta)
    return (slope_length_feet / _REFERENCE_FEET) ** exponent


def steepness_factor(slope_length_feet: float, slope_percent: float) -> float:
    sin_theta = _sin_slope(slope_percent)
    if slope_length_feet < _SHORT_SLOPE_FEET:
        return 3.0 * (sin_theta**0.8) + 0.56
    if slope_percent < 9.0:
        return 10.8 * sin_theta + 0.03
    return 16.8 * sin_theta - 0.50


def optimize_erosion(
    rainfall_erosivity: float,
    soil_erodibility: float,
    slope_length_feet: float,
    slope_percent: float,
    settings: ErosionSettings | None = None,
    tolerance_tons_per_acre_year: float | None = None,
) -> ErosionOptimization:
    """Pick the cover and support pair with the lowest soil loss."""
    catalog = load_erosion_settings() if settings is None else settings
    tolerance = catalog.tolerance_tons_per_acre_year
    if tolerance_tons_per_acre_year is not None:
        _positive_finite("tolerance", tolerance_tons_per_acre_year)
        tolerance = tolerance_tons_per_acre_year
    _percent("slope", slope_percent)
    _positive_finite("slope length", slope_length_feet)
    ranked: list[ErosionChoice] = []
    for cover in catalog.covers:
        for support in catalog.supports:
            factor = support_factor(support, slope_percent, slope_length_feet)
            if factor is None:
                continue
            loss = annual_soil_loss(
                rainfall_erosivity,
                soil_erodibility,
                slope_length_feet,
                slope_percent,
                cover.factor,
                factor,
            )
            ranked.append(
                ErosionChoice(
                    cover=cover.name,
                    support=support.name,
                    tons_per_acre_year=loss.tons_per_acre_year,
                    length_factor=loss.length_factor,
                    steepness_factor=loss.steepness_factor,
                    cover_factor=cover.factor,
                    practice_factor=factor,
                    meets_tolerance=loss.tons_per_acre_year <= tolerance,
                    tolerance_tons_per_acre_year=tolerance,
                )
            )
    if not ranked:
        raise ErosionError("no support practice in the catalog applies to this slope")
    ranked.sort(key=lambda choice: (choice.tons_per_acre_year, choice.cover, choice.support))
    ordered = tuple(ranked)
    return ErosionOptimization(chosen=ordered[0], ranked=ordered)


def rainfall_to_us_customary(value: float, units: str) -> float:
    _positive_finite("rainfall erosivity", value)
    if units == "us-customary":
        return value
    if units == "si":
        return value / _R_SI_PER_US
    raise ErosionError("rainfall erosivity needs a unit system")


def erodibility_to_us_customary(value: float, units: str) -> float:
    _positive_finite("soil erodibility", value)
    if units == "us-customary":
        return value
    if units == "si":
        return value / _K_SI_PER_US
    raise ErosionError("soil erodibility needs a unit system")


def tons_per_acre_to_reported(tons_per_acre: float, units: str) -> float:
    _positive_finite("soil loss", tons_per_acre)
    if units == "us-customary":
        return tons_per_acre
    if units == "si":
        return tons_per_acre * _TONNES_PER_HECTARE_PER_TON_PER_ACRE
    raise ErosionError("soil loss needs a unit system")


def support_factor(
    support: SupportPractice,
    slope_percent: float,
    slope_length_feet: float,
) -> float | None:
    slope_class = math.floor(slope_percent + 0.5)
    length_meters = slope_length_feet * _FEET_TO_METERS
    for band in support.bands:
        if band.low_percent <= slope_class <= band.high_percent:
            if band.max_length_meters > 0 and length_meters > band.max_length_meters:
                return None
            return band.factor
    return None


def _sin_slope(slope_percent: float) -> float:
    return math.sin(math.atan(slope_percent / 100.0))


def _positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ErosionError(f"{name} must be a positive finite number")


def _percent(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0 or value > 300:
        raise ErosionError(f"{name} percent must be from 0 through 300")


def _factor(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0 or value > 1:
        raise ErosionError(f"{name} must be greater than 0 and at most 1")
