from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform

from farm_slm.config import load_map_settings
from farm_slm.maps import MapSheetError, locate_coordinate, write_coordinate_example


class MapTests(unittest.TestCase):
    def test_chip_size_comes_from_the_project_file(self) -> None:
        self.assertEqual(load_map_settings().chip_size, 256)

    def test_longitude_latitude_lands_on_the_hand_computed_pixel(self) -> None:
        with TemporaryDirectory() as directory:
            sheet = Path(directory) / "topo.tif"
            _write_sheet(sheet, crs="EPSG:4326", west=-105.0, north=40.0, size=0.01)
            # Upper-left corner of the sheet is lon -105, lat 40.
            # Pixel column 5, row 2 has its center half a pixel in from there.
            longitude = -105.0 + (5 + 0.5) * 0.01
            latitude = 40.0 - (2 + 0.5) * 0.01
            located = locate_coordinate(sheet, longitude, latitude)
        self.assertEqual(located["column"], 5)
        self.assertEqual(located["row"], 2)

    def test_projected_sheet_round_trips_through_a_chip(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sheet = root / "utm.tif"
            transform_grid = from_origin(500000.0, 4_430000.0, 10.0, 10.0)
            _write_grid(sheet, "EPSG:32613", transform_grid, width=40, height=40)
            with rasterio.open(sheet) as source:
                easting, northing = source.xy(11, 17, offset="center")
                longitude, latitude = transform(source.crs, "EPSG:4326", [easting], [northing])
            record = write_coordinate_example(
                sheet,
                float(longitude[0]),
                float(latitude[0]),
                root / "chips",
                license_id="public-domain",
                chip_size=16,
            )
            self.assertEqual(record["sheet_column"], 17)
            self.assertEqual(record["sheet_row"], 11)
            chip = root / "chips" / str(record["chip"])
            again = locate_coordinate(
                chip,
                float(longitude[0]),
                float(latitude[0]),
            )
        self.assertEqual(again["column"], record["chip_column"])
        self.assertEqual(again["row"], record["chip_row"])
        payload = json.dumps(record)
        self.assertNotIn("\n", payload)
        self.assertIn("chip column", str(record["output"]))

    def test_screenshot_without_a_coordinate_system_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            sheet = Path(directory) / "screenshot.tif"
            _write_grid(sheet, None, from_origin(0.0, 0.0, 1.0, 1.0), width=8, height=8)
            with self.assertRaises(MapSheetError):
                locate_coordinate(sheet, -105.0, 40.0)

    def test_coordinate_outside_the_sheet_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            sheet = Path(directory) / "topo.tif"
            _write_sheet(sheet, crs="EPSG:4326", west=-105.0, north=40.0, size=0.01)
            with self.assertRaises(MapSheetError):
                locate_coordinate(sheet, 10.0, 10.0)


def _write_sheet(path: Path, crs: str, west: float, north: float, size: float) -> None:
    _write_grid(path, crs, from_origin(west, north, size, size), width=20, height=20)


def _write_grid(path: Path, crs: str | None, grid, width: int, height: int) -> None:
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "transform": grid,
    }
    if crs is not None:
        profile["crs"] = crs
    with rasterio.open(path, "w", **profile) as sheet:
        sheet.write(np.zeros((1, height, width), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
