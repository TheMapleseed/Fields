import unittest

from farm_slm.erosion import rainfall_to_us_customary
from farm_slm.locality import (
    LocalityError,
    condition_text,
    erosion_training_record,
    hold_out_region,
)


class LocalityTests(unittest.TestCase):
    def test_si_rainfall_uses_the_handbook_factor(self) -> None:
        self.assertAlmostEqual(rainfall_to_us_customary(1702.0, "si"), 100.0, places=6)
        self.assertEqual(rainfall_to_us_customary(100.0, "us-customary"), 100.0)

    def test_training_text_names_the_place_and_the_units(self) -> None:
        text = condition_text("Millet follows the rains.", "ke", "si")
        self.assertEqual(text, "Place: ke. Units: SI. Millet follows the rains.")
        self.assertEqual(condition_text("Loam holds water.", "unspecified", "unspecified"), "Loam holds water.")

    def test_erosion_example_keeps_si_numbers_and_a_place(self) -> None:
        record = erosion_training_record(
            domain="physics",
            region="KE",
            units="si",
            rainfall_erosivity=1702.0,
            soil_erodibility=0.042,
            slope_length_feet=72.6,
            slope_percent=5,
            license_id="CC-BY-4.0",
            source_url="https://doi.org/10.5281/zenodo.8036998",
        )
        self.assertEqual(record["region"], "ke")
        self.assertEqual(record["units"], "si")
        self.assertIn("Place: ke.", record["text"])
        self.assertIn("Units: SI.", record["text"])
        self.assertIn("1702.0", record["text"])
        self.assertIn("tonnes per hectare per year", record["text"])
        self.assertNotIn("\n", record["text"])

    def test_an_erosion_example_requires_a_place(self) -> None:
        with self.assertRaises(LocalityError):
            erosion_training_record(
                domain="physics",
                region="unspecified",
                units="si",
                rainfall_erosivity=100.0,
                soil_erodibility=0.3,
                slope_length_feet=72.6,
                slope_percent=5,
                license_id="CC-BY-4.0",
                source_url="https://example.test/soils",
            )

    def test_holdout_removes_one_region(self) -> None:
        records = (
            {"region": "ke", "text": "Kenya"},
            {"region": "in", "text": "India"},
        )
        train, held = hold_out_region(records, "in")
        self.assertEqual(tuple(item["region"] for item in train), ("ke",))
        self.assertEqual(tuple(item["region"] for item in held), ("in",))
