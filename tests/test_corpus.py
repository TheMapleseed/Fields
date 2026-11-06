import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from farm_slm.corpus import example_from_record, load_training_examples, resolve_domain
from farm_slm.locality import condition_text


class CorpusTests(unittest.TestCase):
    def test_multiline_text_is_flattened(self) -> None:
        text = condition_text("Line one.\nLine two.", "unspecified", "unspecified")
        self.assertEqual(text, "Line one. Line two.")

    def test_catalog_domains_map_onto_model_domains(self) -> None:
        allowed = (
            "english",
            "agricultural_linguistics",
            "chemistry",
            "weather_reading",
            "physics",
        )
        self.assertEqual(resolve_domain("spatial", allowed), "physics")
        self.assertEqual(resolve_domain("soil", allowed), "chemistry")
        self.assertEqual(resolve_domain("agriculture", allowed), "agricultural_linguistics")
        self.assertEqual(resolve_domain("economics", allowed), "agricultural_linguistics")
        self.assertEqual(resolve_domain("biology", allowed), "chemistry")
        self.assertEqual(resolve_domain("mathematics", allowed), "physics")

    def test_map_record_becomes_a_training_example(self) -> None:
        row = {
            "domain": "spatial",
            "instruction": "Plot this WGS84 coordinate on the topographic sheet.",
            "input": "longitude -105.00000000 latitude 40.00000000",
            "output": "chip column 8 row 8",
            "region": "unspecified",
            "units": "unspecified",
        }
        example = example_from_record(
            row,
            (
                "english",
                "agricultural_linguistics",
                "chemistry",
                "weather_reading",
                "physics",
            ),
        )
        self.assertIsNotNone(example)
        assert example is not None
        self.assertEqual(example.domain, "physics")
        self.assertIn("Instruction:", example.text)
        self.assertIn("Output:", example.text)

    def test_hold_out_splits_loaded_examples(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "data" / "corpus"
            corpus.mkdir(parents=True)
            lines = [
                {"domain": "chemistry", "text": "Kenya soil.", "region": "ke", "units": "si"},
                {"domain": "chemistry", "text": "India soil.", "region": "in", "units": "si"},
            ]
            (corpus / "chemistry.jsonl").write_text(
                "\n".join(json.dumps(row) for row in lines) + "\n",
                encoding="utf-8",
            )
            train, held = load_training_examples(
                root,
                (
                    "english",
                    "agricultural_linguistics",
                    "chemistry",
                    "weather_reading",
                    "physics",
                ),
                hold_out="in",
            )
            self.assertEqual(len(train), 1)
            self.assertEqual(train[0].region, "ke")
            self.assertEqual(len(held), 1)
            self.assertEqual(held[0].region, "in")
