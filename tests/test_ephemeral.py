import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from farm_slm.ephemeral import (
    build_ephemeral_jobs,
    clean_ephemeral,
    ephemeral_directory,
    load_english_seeds,
    make_step_puller,
    pull_training_example,
)
from farm_slm.fetch import primary_plan
from farm_slm.sources import SourceCatalog, SourceItem, load_source_catalog


class EphemeralTests(unittest.TestCase):
    def test_seeds_load(self) -> None:
        words = load_english_seeds()
        self.assertIn("soil", words)
        self.assertIn("tilth", words)
        self.assertGreater(len(words), 20)

    def test_dictionary_catalog_root_is_seed_expanded(self) -> None:
        catalog = load_source_catalog()
        dictionary_items = [item for item in catalog.items if item.kind == "english-dictionary"]
        self.assertEqual(len(dictionary_items), 1)
        self.assertEqual(primary_plan(dictionary_items[0]).channel, "dictionary")
        self.assertFalse(primary_plan(dictionary_items[0]).fetchable)
        jobs = build_ephemeral_jobs(catalog, seeds=("soil", "tilth"))
        dictionary_jobs = [job for job in jobs if job.channel == "dictionary"]
        self.assertEqual(len(dictionary_jobs), 2)
        self.assertTrue(dictionary_jobs[0].url.endswith("/soil"))

    def test_pull_cleans_staging_when_ephemeral(self) -> None:
        item = SourceItem(
            source_id="dictionaryapi-dev",
            source_title="Free Dictionary API",
            title="soil",
            domain="english",
            model_domain="english",
            url="https://api.dictionaryapi.dev/api/v2/entries/en/soil",
            license="CC-BY-SA",
            covers="definitions",
            kind="english-dictionary",
        )
        job = build_ephemeral_jobs(SourceCatalog((item,)), seeds=("soil",))[0]

        def transport(url: str, headers: dict[str, str]) -> bytes:
            self.assertTrue(url.endswith("/soil"))
            return json.dumps(
                [
                    {
                        "word": "soil",
                        "meanings": [
                            {
                                "partOfSpeech": "noun",
                                "definitions": [
                                    {
                                        "definition": "the upper layer of earth in which plants grow",
                                        "example": "sandy soil",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            ).encode("utf-8")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            from farm_slm.config import CollectionSettings

            settings = CollectionSettings(
                max_repos_per_query=1,
                max_files_per_repo=1,
                max_file_bytes=100_000,
                max_total_bytes=100_000,
                max_queries_per_domain=1,
                fetch_catalog=False,
                fetch_during_train=True,
                ephemeral_corpus=True,
                ephemeral_items_per_step=1,
                pdf_max_bytes=5_000_000,
                pdf_chars_per_page=4000,
            )
            example = pull_training_example(
                job,
                ("english",),
                transport=transport,
                root=root,
                settings=settings,
            )
            self.assertEqual(example.domain, "english")
            self.assertIn("plants grow", example.text)
            self.assertFalse(ephemeral_directory(root).exists())

    def test_step_puller_falls_back_on_fetch_error(self) -> None:
        item = SourceItem(
            source_id="dictionaryapi-dev",
            source_title="Free Dictionary API",
            title="soil",
            domain="english",
            model_domain="english",
            url="https://api.dictionaryapi.dev/api/v2/entries/en/",
            license="CC-BY-SA",
            covers="definitions",
            kind="english-dictionary",
        )
        jobs = build_ephemeral_jobs(SourceCatalog((item,)), seeds=("soil",))

        def transport(url: str, headers: dict[str, str]) -> bytes:
            raise OSError("offline")

        from farm_slm.config import CollectionSettings

        settings = CollectionSettings(
            max_repos_per_query=1,
            max_files_per_repo=1,
            max_file_bytes=100_000,
            max_total_bytes=100_000,
            max_queries_per_domain=1,
            fetch_catalog=False,
            fetch_during_train=True,
            ephemeral_corpus=True,
            ephemeral_items_per_step=1,
            pdf_max_bytes=5_000_000,
            pdf_chars_per_page=4000,
        )
        puller = make_step_puller(
            jobs,
            ("english",),
            items_per_step=1,
            transport=transport,
            settings=settings,
        )
        self.assertIsNone(puller(1))
        clean_ephemeral()


if __name__ == "__main__":
    unittest.main()
