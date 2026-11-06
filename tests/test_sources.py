import unittest

from farm_slm.collect import catalog_plan, collection_section, empty_plan
from farm_slm.config import load_project
from farm_slm.dumps import catalog_dump_plan
from farm_slm.fetch import primary_plan
from farm_slm.maps import catalog_map_plan
from farm_slm.sources import (
    dump_targets_from_catalog,
    github_plan_from_catalog,
    load_source_catalog,
    map_targets_from_catalog,
)


class SourcesTests(unittest.TestCase):
    def test_catalog_loads_and_maps_domains(self) -> None:
        catalog = load_source_catalog()
        self.assertGreater(len(catalog.items), 20)
        domains = {item.model_domain for item in catalog.items}
        self.assertTrue(domains.issubset(
            {
                "english",
                "agricultural_linguistics",
                "chemistry",
                "weather_reading",
                "physics",
            }
        ))
        self.assertTrue(all(item.url.startswith("https://") for item in catalog.items))

    def test_github_plan_comes_from_the_catalog_without_network(self) -> None:
        plan = catalog_plan()
        self.assertEqual(set(plan), set(empty_plan()))
        self.assertTrue(any(plan[domain] for domain in plan))
        for queries in plan.values():
            for query in queries:
                self.assertIn("agriculture", query)

    def test_collection_section_lists_catalog_targets(self) -> None:
        section = collection_section(empty_plan())
        self.assertIn("Source catalog targets", section)
        self.assertIn("fetch=off", section)
        self.assertIn("Catalog fetch routing", section)
        self.assertIn("Deferred pipelines", section)
        self.assertIn("Map sheet intake", section)
        self.assertIn("Dump and document intake", section)
        self.assertIn("items=", section)
        self.assertFalse(load_project().collection.fetch_catalog)

    def test_plan_from_catalog_matches_loader(self) -> None:
        catalog = load_source_catalog()
        self.assertEqual(github_plan_from_catalog(catalog), catalog_plan())

    def test_deferred_pipelines_split_maps_and_dumps(self) -> None:
        catalog = load_source_catalog()
        maps = map_targets_from_catalog(catalog)
        dumps = dump_targets_from_catalog(catalog)
        self.assertEqual(len(maps), 8)
        # Wikipedia dump stays deferred; RUSLE PDF moved to the pdf channel.
        self.assertEqual(len(dumps), 1)
        self.assertTrue(all(item.intake == "maps" for item in maps))
        self.assertTrue(all(item.channel.startswith("deferred_") for item in dumps))
        self.assertEqual(catalog_map_plan(catalog), maps)
        self.assertEqual(catalog_dump_plan(catalog).targets, dumps)
        pdf_items = [item for item in catalog.items if primary_plan(item).channel == "pdf"]
        self.assertGreaterEqual(len(pdf_items), 10)
