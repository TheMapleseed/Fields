import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from farm_slm.ephemeral import EphemeralCursor, EphemeralJob
from farm_slm.config import CollectionSettings
from farm_slm.stage import StageState, jobs_fingerprint, load_stage, save_stage


def _settings() -> CollectionSettings:
    return CollectionSettings(
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


class StageTests(unittest.TestCase):
    def test_stage_round_trips(self) -> None:
        state = StageState(
            step=12,
            job_index=3,
            page_index=7,
            job_url="https://cdn.mises.org/principles_of_economics.pdf",
            job_title="Principles of Economics (Menger)",
            jobs_sha="abc123",
            order=(2, 0, 1),
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = save_stage(root, state)
            self.assertTrue(path.is_file())
            loaded = load_stage(root)
        self.assertEqual(loaded, state)

    def test_cursor_restore_keeps_page(self) -> None:
        jobs = (
            EphemeralJob(
                title="A",
                model_domain="chemistry",
                url="https://example.com/a.pdf",
                license="CC-BY",
                source_id="a",
                kind="soil",
                covers="",
                channel="pdf",
            ),
            EphemeralJob(
                title="B",
                model_domain="chemistry",
                url="https://example.com/b.pdf",
                license="CC-BY",
                source_id="b",
                kind="soil",
                covers="",
                channel="pdf",
            ),
        )
        cursor = EphemeralCursor(
            jobs=jobs,
            domains=("chemistry",),
            transport=None,
            root=None,
            settings=_settings(),
        )
        cursor.restore(1, 9, "https://example.com/b.pdf")
        self.assertEqual(cursor.job_index, 1)
        self.assertEqual(cursor.page_index, 9)
        self.assertEqual(cursor.current_job().url, "https://example.com/b.pdf")
        self.assertEqual(jobs_fingerprint(jobs), jobs_fingerprint(list(jobs)))


if __name__ == "__main__":
    unittest.main()
