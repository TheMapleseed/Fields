import json
import tempfile
import unittest
from pathlib import Path

from farm_slm.collect import (
    CollectionError,
    _github_url,
    collect_github,
    empty_plan,
    select_text_blobs,
    text_from_bytes,
)
from farm_slm.config import load_project
from farm_slm.kdl import parse_kdl


class ConfigTests(unittest.TestCase):
    def test_shipped_context_window_and_weights(self) -> None:
        project = load_project()
        self.assertEqual(project.model.max_position_embeddings, 500_000)
        self.assertEqual(project.model.segment_length, 2048)
        self.assertEqual(project.model.memory_slots, 256)
        self.assertEqual(
            project.model.domains,
            (
                "english",
                "agricultural_linguistics",
                "chemistry",
                "weather_reading",
                "physics",
            ),
        )
        self.assertEqual(project.weights_format, "safetensors")
        self.assertTrue(project.weights_path.endswith(".safetensors"))

    def test_kdl_comments_and_numbers(self) -> None:
        nodes = parse_kdl('model { rms_norm_eps 1e-6 }\n')
        self.assertEqual(nodes[0].children[0].args[0], 1e-6)


class CollectTests(unittest.TestCase):
    def test_empty_plan_does_not_call_github(self) -> None:
        def transport(url: str, headers: dict[str, str]) -> dict[str, object]:
            raise AssertionError(url)

        with tempfile.TemporaryDirectory() as directory:
            summary = collect_github(
                empty_plan(),
                root=Path(directory),
                token="",
                transport=transport,
            )
        self.assertEqual(summary.written, 0)
        self.assertIn("No GitHub queries", summary.note)

    def test_selects_prose_and_skips_secrets(self) -> None:
        tree = [
            {"path": "docs/soil.md", "type": "blob", "sha": "a" * 40, "size": 80},
            {"path": ".env", "type": "blob", "sha": "b" * 40, "size": 20},
            {"path": "keys/id_rsa", "type": "blob", "sha": "c" * 40, "size": 20},
            {"path": "vendor/readme.md", "type": "blob", "sha": "d" * 40, "size": 20},
            {"path": "../escape.md", "type": "blob", "sha": "e" * 40, "size": 20},
        ]
        selected = select_text_blobs(tree, max_file_bytes=1000, max_files=10)
        self.assertEqual(selected, [("docs/soil.md", "a" * 40, 80)])

    def test_rejects_private_key_text(self) -> None:
        raw = b"notes\n-----BEGIN RSA PRIVATE KEY-----\nabc\n"
        self.assertIsNone(text_from_bytes("docs/notes.md", raw))

    def test_refuses_off_api_paths(self) -> None:
        with self.assertRaises(CollectionError):
            _github_url("//evil.example/steal")

    def test_collects_one_markdown_file(self) -> None:
        soil = b"# Soil\n\nLoam holds water and nutrients for the crop.\n"
        secret = b"# Notes\n\n-----BEGIN RSA PRIVATE KEY-----\nsecret\n"
        sha_soil = "a" * 40
        sha_secret = "c" * 40

        def transport(url: str, headers: dict[str, str]) -> dict[str, object]:
            self.assertTrue(url.startswith("https://api.github.com/"))
            self.assertEqual(headers.get("Authorization"), "Bearer test-token")
            if "/search/repositories?" in url:
                return {
                    "items": [
                        {
                            "full_name": "example/soils",
                            "default_branch": "main",
                            "html_url": "https://github.com/example/soils",
                            "license": {"spdx_id": "MIT"},
                        }
                    ]
                }
            if "/git/trees/" in url:
                return {
                    "truncated": False,
                    "tree": [
                        {"path": "docs/soil.md", "type": "blob", "sha": sha_soil, "size": len(soil)},
                        {"path": ".env", "type": "blob", "sha": "b" * 40, "size": 12},
                        {
                            "path": "docs/keys.md",
                            "type": "blob",
                            "sha": sha_secret,
                            "size": len(secret),
                        },
                    ],
                }
            if url.endswith("/git/blobs/" + sha_soil):
                return {"encoding": "base64", "content": _b64(soil)}
            if url.endswith("/git/blobs/" + sha_secret):
                return {"encoding": "base64", "content": _b64(secret)}
            raise AssertionError(url)

        plan = empty_plan()
        plan["chemistry"] = ["soil chemistry"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = collect_github(plan, root=root, token="test-token", transport=transport)
            written = (root / "data" / "corpus" / "chemistry.jsonl").read_text(encoding="utf-8")
        self.assertEqual(summary.written, 1)
        record = json.loads(written)
        self.assertEqual(record["domain"], "chemistry")
        self.assertEqual(record["repo"], "example/soils")
        self.assertEqual(record["path"], "docs/soil.md")
        self.assertEqual(record["license"], "MIT")
        self.assertIn("Loam holds water", record["text"])
        self.assertNotIn("PRIVATE KEY", written)
        self.assertTrue(record["source_url"].startswith("https://github.com/example/soils/blob/"))


def _b64(raw: bytes) -> str:
    import base64

    return base64.b64encode(raw).decode("ascii")


if __name__ == "__main__":
    unittest.main()
