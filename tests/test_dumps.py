import unittest
from pathlib import Path

from farm_slm.dumps import DumpError, catalog_dump_plan, dump_intake_report, read_dump_text


class DumpTests(unittest.TestCase):
    def test_catalog_dump_plan_lists_deferred_targets(self) -> None:
        plan = catalog_dump_plan()
        self.assertEqual(len(plan.targets), 1)
        self.assertTrue(plan.staging_dir.as_posix().endswith("data/dumps"))
        report = dump_intake_report()
        self.assertIn("Dump and document intake", report)
        self.assertIn("deferred_dump", report)

    def test_read_dump_text_stays_deferred(self) -> None:
        with self.assertRaises(DumpError):
            read_dump_text(Path("data/dumps/example.xml.bz2"))


if __name__ == "__main__":
    unittest.main()
