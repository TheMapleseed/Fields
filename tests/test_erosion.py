import math
import unittest

from farm_slm.config import load_erosion_settings
from farm_slm.erosion import ErosionError, annual_soil_loss, length_factor, optimize_erosion


class ErosionTests(unittest.TestCase):
    def test_reference_plot_length_and_five_percent_steepness(self) -> None:
        self.assertEqual(length_factor(72.6, 5), 1.0)
        sin_theta = math.sin(math.atan(0.05))
        steepness = 10.8 * sin_theta + 0.03
        loss = annual_soil_loss(150, 0.32, 72.6, 5, 0.32, 0.50)
        self.assertAlmostEqual(loss.steepness_factor, steepness, places=9)
        self.assertAlmostEqual(loss.tons_per_acre_year, 150 * 0.32 * steepness * 0.32 * 0.50, places=9)

    def test_catalog_uses_the_contour_table(self) -> None:
        settings = load_erosion_settings()
        self.assertEqual(settings.tolerance_tons_per_acre_year, 5)
        contour = next(item for item in settings.supports if item.name == "contour")
        five_percent = next(band for band in contour.bands if band.low_percent == 3)
        self.assertEqual(five_percent.factor, 0.50)
        self.assertEqual(five_percent.max_length_meters, 90)

    def test_optimization_picks_the_lowest_soil_loss(self) -> None:
        result = optimize_erosion(150, 0.32, 72.6, 5)
        self.assertEqual(result.chosen.cover, "permanent-grass")
        self.assertEqual(result.chosen.support, "contour-strip-crop")
        self.assertTrue(result.chosen.meets_tolerance)
        self.assertLess(result.chosen.tons_per_acre_year, result.ranked[-1].tons_per_acre_year)
        bare = next(item for item in result.ranked if item.cover == "bare-fallow" and item.support == "up-and-down-slope")
        self.assertGreater(bare.tons_per_acre_year, result.chosen.tons_per_acre_year)

    def test_long_slope_drops_contouring(self) -> None:
        result = optimize_erosion(150, 0.32, 1000, 5)
        supports = {item.support for item in result.ranked}
        self.assertEqual(supports, {"up-and-down-slope"})

    def test_unmet_tolerance_is_reported_on_the_best_plan(self) -> None:
        result = optimize_erosion(150, 0.32, 72.6, 5, tolerance_tons_per_acre_year=0.0001)
        self.assertFalse(result.chosen.meets_tolerance)
        self.assertEqual(result.chosen.cover, "permanent-grass")

    def test_bad_inputs_are_rejected(self) -> None:
        with self.assertRaises(ErosionError):
            annual_soil_loss(150, 0.32, 72.6, 5, 0, 1)
        with self.assertRaises(ErosionError):
            annual_soil_loss(float("nan"), 0.32, 72.6, 5, 0.3, 1)
