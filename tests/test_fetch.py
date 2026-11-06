import json
import tempfile
import unittest
from pathlib import Path

from farm_slm.collect import collect_catalog
from farm_slm.fetch import fetch_item, primary_plan, route_item, routing_report
from farm_slm.sources import SourceCatalog, SourceItem, load_source_catalog


def _item(
    *,
    title: str,
    url: str,
    kind: str = "agriculture",
    domain: str = "agricultural_linguistics",
) -> SourceItem:
    return SourceItem(
        source_id="test",
        source_title="Test",
        title=title,
        domain=domain,
        model_domain=domain,
        url=url,
        license="CC-BY-4.0",
        covers="test",
        kind=kind,
    )


class FetchRoutingTests(unittest.TestCase):
    def test_every_catalog_item_has_a_primary_channel(self) -> None:
        catalog = load_source_catalog()
        channels = {primary_plan(item).channel for item in catalog.items}
        self.assertNotIn("unrouted", channels)
        report = routing_report(catalog.items)
        self.assertIn("Catalog fetch routing", report)
        self.assertIn(f"items={len(catalog.items)}", report)
        for item in catalog.items:
            plans = route_item(item)
            self.assertEqual(plans[-1].channel, "github_slot")

    def test_wikipedia_html_json_and_deferred_routes(self) -> None:
        dictionary = primary_plan(
            _item(
                title="dictionaryapi.dev English entries",
                url="https://api.dictionaryapi.dev/api/v2/entries/en/",
                kind="english-dictionary",
                domain="english",
            )
        )
        self.assertEqual(dictionary.channel, "dictionary")
        self.assertFalse(dictionary.fetchable)
        self.assertEqual(
            primary_plan(
                _item(
                    title="soil",
                    url="https://api.dictionaryapi.dev/api/v2/entries/en/soil",
                    kind="english-dictionary",
                    domain="english",
                )
            ).channel,
            "dictionary",
        )
        self.assertEqual(
            primary_plan(
                _item(
                    title="Wiki",
                    url="https://en.wikipedia.org/wiki/Agriculture",
                )
            ).channel,
            "wikipedia",
        )
        self.assertEqual(
            primary_plan(
                _item(
                    title="Portal",
                    url="https://www.fao.org/documents",
                )
            ).channel,
            "html",
        )
        self.assertEqual(
            primary_plan(
                _item(
                    title="IEM",
                    url="https://mesonet.agron.iastate.edu/geojson/nwstext.py?help=",
                    kind="weather",
                    domain="weather_reading",
                )
            ).channel,
            "json_document",
        )
        self.assertEqual(
            primary_plan(
                _item(
                    title="DEM",
                    url="https://portal.opentopography.org/dataCatalog",
                    kind="elevation",
                    domain="physics",
                )
            ).channel,
            "deferred_geospatial",
        )
        self.assertEqual(
            primary_plan(
                _item(
                    title="Dump",
                    url="https://dumps.wikimedia.org/",
                )
            ).channel,
            "deferred_dump",
        )
        self.assertEqual(
            primary_plan(
                _item(
                    title="PDF",
                    url="https://sis.agr.gc.ca/cansis/publications/manuals/2002-92/rusle-can.pdf",
                    kind="soil",
                    domain="chemistry",
                )
            ).channel,
            "pdf",
        )
        self.assertEqual(
            primary_plan(
                _item(
                    title="Chemistry 2e",
                    url="https://openstax.org/details/books/chemistry-2e",
                    kind="textbooks",
                    domain="chemistry",
                )
            ).channel,
            "pdf",
        )

    def test_collect_catalog_dry_run_does_not_hit_the_network(self) -> None:
        def transport(url: str, headers: dict[str, str]) -> bytes:
            raise AssertionError(url)

        summary = collect_catalog(fetch=False, transport=transport)
        self.assertEqual(summary.written, 0)
        self.assertIn("Catalog dry-run", summary.note)
        self.assertIn("Catalog fetch routing", summary.note)

    def test_collect_catalog_fetch_writes_jsonl_with_license(self) -> None:
        item = _item(
            title="Plain Language",
            url="https://www.plainlanguage.gov/",
            kind="english",
            domain="english",
        )

        def transport(url: str, headers: dict[str, str]) -> bytes:
            self.assertEqual(url, item.url)
            return b"<html><body><p>Clear writing helps farmers follow the guidance.</p></body></html>"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = collect_catalog(
                fetch=True,
                root=root,
                transport=transport,
                catalog=SourceCatalog((item,)),
            )
            path = root / "data" / "corpus" / "english.catalog.jsonl"
            self.assertEqual(summary.written, 1)
            self.assertTrue(path.is_file())
            row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["license"], "CC-BY-4.0")
            self.assertEqual(row["channel"], "html")
            self.assertIn("farmers", row["text"])

    def test_fetch_item_uses_wikipedia_api_shape(self) -> None:
        item = _item(
            title="Agriculture",
            url="https://en.wikipedia.org/wiki/Agriculture",
        )

        def transport(url: str, headers: dict[str, str]) -> bytes:
            self.assertIn("api.php", url)
            return json.dumps(
                {
                    "query": {
                        "pages": {
                            "1": {"extract": "Agriculture is the practice of cultivating plants."}
                        }
                    }
                }
            ).encode("utf-8")

        result = fetch_item(item, transport)
        self.assertEqual(result.content_type, "wikipedia-extract")
        self.assertIn("cultivating", result.text)


if __name__ == "__main__":
    unittest.main()
