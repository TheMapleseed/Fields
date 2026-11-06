import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from farm_slm.config import CollectionSettings
from farm_slm.ephemeral import build_ephemeral_jobs, make_step_puller
from farm_slm.pdf import PdfError, PdfSession, extract_page_text, resolve_pdf_url
from farm_slm.sources import SourceCatalog, SourceItem


def _pdf_with_pages(*texts: str) -> bytes:
    writer = PdfWriter()
    for text in texts:
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        page = writer.add_blank_page(width=400, height=200)
        content = f"BT /F1 12 Tf 40 120 Td ({safe}) Tj ET"
        stream = DecodedStreamObject()
        stream.set_data(content.encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {
                        NameObject("/F1"): DictionaryObject(
                            {
                                NameObject("/Type"): NameObject("/Font"),
                                NameObject("/Subtype"): NameObject("/Type1"),
                                NameObject("/BaseFont"): NameObject("/Helvetica"),
                            }
                        )
                    }
                )
            }
        )
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class PdfTests(unittest.TestCase):
    def test_extract_page_text_from_bytes(self) -> None:
        data = _pdf_with_pages(
            "Nitrogen is a chemical element essential to plant growth.",
            "Phosphorus supports root development in soils.",
        )
        first = extract_page_text(data, 0, max_chars=4000)
        self.assertEqual(first.page_index, 0)
        self.assertEqual(first.page_count, 2)
        self.assertIn("Nitrogen", first.text)
        second = extract_page_text(data, 1, max_chars=4000)
        self.assertIn("Phosphorus", second.text)

    def test_resolve_openstax_pdf_url(self) -> None:
        def transport(url: str, headers: dict[str, str]) -> bytes:
            self.assertIn("chemistry-2e", url)
            return json.dumps(
                {
                    "items": [
                        {
                            "title": "Chemistry 2e",
                            "pdf_url": (
                                "https://assets.openstax.org/oscms-prodcms/media/"
                                "documents/chemistry-2e_-_WEB.pdf"
                            ),
                        }
                    ]
                }
            ).encode("utf-8")

        resolved = resolve_pdf_url(
            "https://openstax.org/details/books/chemistry-2e",
            transport,
        )
        self.assertTrue(resolved.endswith("chemistry-2e_-_WEB.pdf"))

    def test_session_walks_pages_then_deletes_cache(self) -> None:
        pdf_bytes = _pdf_with_pages(
            "Page one discusses soil chemistry and pH buffers carefully.",
            "Page two covers cation exchange in agricultural soils today.",
        )

        def transport(url: str, headers: dict[str, str]) -> bytes:
            if "api/v2/pages" in url:
                return json.dumps(
                    {
                        "items": [
                            {
                                "pdf_url": "https://assets.openstax.org/example/chemistry.pdf",
                            }
                        ]
                    }
                ).encode("utf-8")
            if url.endswith(".pdf"):
                return pdf_bytes
            raise AssertionError(url)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = PdfSession(
                root=root,
                max_bytes=5_000_000,
                max_chars=4000,
                transport=transport,
            )
            first, finished = session.pull_page(
                "https://openstax.org/details/books/chemistry-2e",
                0,
            )
            self.assertFalse(finished)
            self.assertIn("soil chemistry", first.text)
            self.assertTrue(session.active_path is not None and session.active_path.exists())
            second, finished = session.pull_page(
                "https://openstax.org/details/books/chemistry-2e",
                1,
            )
            self.assertTrue(finished)
            self.assertIn("cation exchange", second.text)
            self.assertIsNone(session.active_path)

    def test_ephemeral_puller_feeds_pdf_pages(self) -> None:
        pdf_bytes = _pdf_with_pages(
            "Osmosis moves water through a partially permeable membrane in cells.",
            "Chlorophyll absorbs light for photosynthesis in green plants.",
        )

        def transport(url: str, headers: dict[str, str]) -> bytes:
            if url.endswith(".pdf"):
                return pdf_bytes
            raise AssertionError(url)

        item = SourceItem(
            source_id="rusle",
            source_title="RUSLE",
            title="RUSLEFAC",
            domain="chemistry",
            model_domain="chemistry",
            url="https://example.com/manual.pdf",
            license="government-publication",
            covers="erosion",
            kind="soil",
        )
        jobs = build_ephemeral_jobs(SourceCatalog((item,)), seeds=())
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].channel, "pdf")
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
        with TemporaryDirectory() as directory:
            root = Path(directory)
            puller = make_step_puller(
                jobs,
                ("chemistry",),
                items_per_step=1,
                transport=transport,
                root=root,
                settings=settings,
            )
            first = puller(1)
            second = puller(2)
            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            assert first is not None and second is not None
            self.assertIn("Osmosis", first.text)
            self.assertIn("Chlorophyll", second.text)

    def test_oversize_pdf_is_rejected(self) -> None:
        def transport(url: str, headers: dict[str, str]) -> bytes:
            return b"%PDF-1.4 oversized-placeholder" + (b"x" * 1_500_000)

        with TemporaryDirectory() as directory:
            session = PdfSession(
                root=Path(directory),
                max_bytes=1_000_000,
                max_chars=4000,
                transport=transport,
            )
            with self.assertRaises(PdfError):
                session.pull_page("https://example.com/big.pdf", 0)


if __name__ == "__main__":
    unittest.main()
