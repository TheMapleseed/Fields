"""Plot a coordinate on a georeferenced topographic sheet.

The pixel label comes from the file's coordinate system and geotransform.
A screenshot has neither, so it is rejected. NAIP foliage and USGS
topographic sheets are the inputs this accepts when they are real GeoTIFFs.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import rasterio
from rasterio.warp import transform
from rasterio.windows import Window

from farm_slm.config import load_map_settings

_STEM = re.compile(r"[^A-Za-z0-9._-]+")


class MapSheetError(ValueError):
    pass


def locate_coordinate(sheet: Path, longitude: float, latitude: float) -> dict[str, object]:
    """Return the pixel that contains this WGS84 coordinate."""
    _validate_wgs84(longitude, latitude)
    with rasterio.open(sheet) as source:
        located = _locate(source, longitude, latitude)
    located["sheet"] = str(sheet)
    return located


def write_coordinate_example(
    sheet: Path,
    longitude: float,
    latitude: float,
    output_dir: Path,
    license_id: str,
    chip_size: int | None = None,
) -> dict[str, object]:
    """Cut a georeferenced chip and record the pixel that holds the coordinate.

    The chip keeps its own geotransform. Opening the chip and plotting the
    same coordinate returns that same pixel.
    """
    _validate_wgs84(longitude, latitude)
    if not isinstance(license_id, str) or not license_id.strip() or any(
        char in license_id for char in "\r\n"
    ):
        raise MapSheetError("a license must be one line of text")
    size = load_map_settings().chip_size if chip_size is None else chip_size
    if size < 8 or size > 2048 or size % 2 != 0:
        raise MapSheetError("chip_size must be an even integer from 8 to 2048")

    output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(sheet) as source:
        located = _locate(source, longitude, latitude)
        window, chip_column, chip_row = _chip_window(
            column=int(located["column"]),
            row=int(located["row"]),
            width=source.width,
            height=source.height,
            chip_size=size,
        )
        samples = source.read(window=window)
        chip_transform = source.window_transform(window)
        profile = {
            "driver": "GTiff",
            "height": samples.shape[1],
            "width": samples.shape[2],
            "count": samples.shape[0],
            "dtype": samples.dtype,
            "crs": source.crs,
            "transform": chip_transform,
            "compress": "deflate",
        }
        chip_name = _chip_name(sheet, int(located["column"]), int(located["row"]))
        chip_path = (output_dir / chip_name).resolve()
        if output_dir.resolve() not in chip_path.parents:
            raise MapSheetError("refusing to write the chip outside the output directory")
        with rasterio.open(chip_path, "w", **profile) as chip:
            chip.write(samples)

    confirmed = locate_coordinate(chip_path, longitude, latitude)
    if int(confirmed["column"]) != chip_column or int(confirmed["row"]) != chip_row:
        chip_path.unlink(missing_ok=True)
        raise MapSheetError("the chip geotransform does not place the coordinate on the labeled pixel")

    record: dict[str, object] = {
        "domain": "spatial",
        "kind": "topographic-coordinate",
        "license": license_id.strip(),
        "sheet": str(sheet),
        "chip": chip_name,
        "crs": str(located["crs"]),
        "longitude": longitude,
        "latitude": latitude,
        "easting": located["easting"],
        "northing": located["northing"],
        "sheet_column": located["column"],
        "sheet_row": located["row"],
        "chip_column": chip_column,
        "chip_row": chip_row,
        "instruction": "Plot this WGS84 coordinate on the topographic sheet.",
        "input": f"longitude {longitude:.8f} latitude {latitude:.8f}",
        "output": (
            f"chip column {chip_column} row {chip_row} "
            f"in {located['crs']} easting {located['easting']:.3f} "
            f"northing {located['northing']:.3f}"
        ),
    }
    return record


def append_example(record: dict[str, object], records_path: Path) -> None:
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    if "\n" in line or "\r" in line:
        raise MapSheetError("record broke the JSONL line")
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _locate(source: rasterio.DatasetReader, longitude: float, latitude: float) -> dict[str, object]:
    if source.crs is None or source.transform.is_identity:
        raise MapSheetError(
            "this image has no map coordinate system, so a coordinate cannot be plotted on it"
        )
    easting, northing = transform("EPSG:4326", source.crs, [longitude], [latitude])
    x_value = float(easting[0])
    y_value = float(northing[0])
    if not math.isfinite(x_value) or not math.isfinite(y_value):
        raise MapSheetError("the coordinate did not project onto this sheet")
    row, column = source.index(x_value, y_value)
    row = int(row)
    column = int(column)
    if row < 0 or column < 0 or row >= source.height or column >= source.width:
        raise MapSheetError("the coordinate is outside this sheet")
    return {
        "crs": source.crs.to_string(),
        "easting": x_value,
        "northing": y_value,
        "column": column,
        "row": row,
    }


def _chip_window(
    column: int,
    row: int,
    width: int,
    height: int,
    chip_size: int,
) -> tuple[Window, int, int]:
    chip_width = min(chip_size, width)
    chip_height = min(chip_size, height)
    column_off = min(max(column - chip_width // 2, 0), width - chip_width)
    row_off = min(max(row - chip_height // 2, 0), height - chip_height)
    return (
        Window(column_off, row_off, chip_width, chip_height),
        column - column_off,
        row - row_off,
    )


def _validate_wgs84(longitude: float, latitude: float) -> None:
    if not math.isfinite(longitude) or not math.isfinite(latitude):
        raise MapSheetError("longitude and latitude must be finite")
    if not -180.0 <= longitude <= 180.0 or not -90.0 <= latitude <= 90.0:
        raise MapSheetError("longitude or latitude is outside WGS84")


def _chip_name(sheet: Path, column: int, row: int) -> str:
    stem = _STEM.sub("-", sheet.stem).strip(".-") or "sheet"
    return f"{stem}_{column}_{row}.tif"
