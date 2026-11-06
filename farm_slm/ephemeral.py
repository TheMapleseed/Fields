"""Pull catalog text for one training step, then discard it.

Training machines that cannot keep a permanent corpus use this path.
Nothing under data/corpus/ is kept across steps when ephemeral_corpus is on.

PDF textbooks (OpenStax and direct .pdf URLs) use PdfSession: one book is
cached under .ephemeral/pdf/, one page is extracted per step, then the file
is deleted when the book is finished or training ends.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from farm_slm.config import CollectionSettings, load_project, project_root
from farm_slm.corpus import TrainingExample, example_from_record
from farm_slm.fetch import FetchError, Transport, fetch_item, primary_plan
from farm_slm.kdl import KdlError, parse_kdl
from farm_slm.pdf import PdfError, PdfSession
from farm_slm.sources import SourceCatalog, SourceItem, SourcesError, load_source_catalog

_WORD = re.compile(r"^[A-Za-z][A-Za-z'-]{0,63}$")
_SEED_FILE = "english_seeds.kdl"


@dataclass(frozen=True)
class EphemeralJob:
    title: str
    model_domain: str
    url: str
    license: str
    source_id: str
    kind: str
    covers: str
    channel: str


class EphemeralError(RuntimeError):
    pass


def ephemeral_directory(root: Path | None = None) -> Path:
    base = project_root() if root is None else root
    return base / "data" / "corpus" / ".ephemeral"


def clean_ephemeral(root: Path | None = None) -> None:
    staging = ephemeral_directory(root)
    if staging.exists():
        shutil.rmtree(staging)


def load_english_seeds(path: Path | None = None) -> tuple[str, ...]:
    seed_path = project_root() / _SEED_FILE if path is None else path
    try:
        text = seed_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EphemeralError(f"cannot read {seed_path}") from exc
    try:
        nodes = parse_kdl(text)
    except KdlError as exc:
        raise EphemeralError(f"invalid english seeds: {exc}") from exc
    words: list[str] = []
    seen: set[str] = set()
    for node in nodes:
        if node.name != "seeds":
            continue
        for child in node.children:
            if child.name != "word":
                continue
            if len(child.args) != 1 or not isinstance(child.args[0], str):
                raise EphemeralError("each seed word must be one string")
            word = child.args[0].strip()
            if not _WORD.fullmatch(word):
                raise EphemeralError(f"refusing seed word {word!r}")
            key = word.lower()
            if key in seen:
                continue
            seen.add(key)
            words.append(word)
    if not words:
        raise EphemeralError("english_seeds.kdl lists no words")
    return tuple(words)


def build_ephemeral_jobs(
    catalog: SourceCatalog | None = None,
    seeds: Sequence[str] | None = None,
) -> tuple[EphemeralJob, ...]:
    current = catalog if catalog is not None else load_source_catalog()
    word_list = tuple(seeds) if seeds is not None else load_english_seeds()
    jobs: list[EphemeralJob] = []
    for item in current.items:
        plan = primary_plan(item)
        if plan.channel == "dictionary":
            jobs.extend(_dictionary_jobs(item, plan.channel, word_list))
            continue
        if plan.channel == "pdf":
            jobs.append(
                EphemeralJob(
                    title=item.title,
                    model_domain=item.model_domain,
                    url=item.url,
                    license=item.license,
                    source_id=item.source_id,
                    kind=item.kind,
                    covers=item.covers,
                    channel=plan.channel,
                )
            )
            continue
        if not plan.fetchable:
            continue
        jobs.append(
            EphemeralJob(
                title=item.title,
                model_domain=item.model_domain,
                url=item.url,
                license=item.license,
                source_id=item.source_id,
                kind=item.kind,
                covers=item.covers,
                channel=plan.channel,
            )
        )
    return tuple(jobs)


def _dictionary_jobs(
    item: SourceItem,
    channel: str,
    word_list: Sequence[str],
) -> tuple[EphemeralJob, ...]:
    from farm_slm.fetch import _dictionary_word

    concrete = _dictionary_word(item.url)
    if concrete is not None:
        return (
            EphemeralJob(
                title=f"{item.title}:{concrete}",
                model_domain=item.model_domain,
                url=item.url,
                license=item.license,
                source_id=item.source_id,
                kind=item.kind,
                covers=item.covers,
                channel=channel,
            ),
        )
    base = item.url if item.url.endswith("/") else f"{item.url}/"
    return tuple(
        EphemeralJob(
            title=f"{item.title}:{word}",
            model_domain=item.model_domain,
            url=f"{base}{word}",
            license=item.license,
            source_id=item.source_id,
            kind=item.kind,
            covers=item.covers,
            channel=channel,
        )
        for word in word_list
    )


def pull_training_example(
    job: EphemeralJob,
    domains: Sequence[str],
    *,
    transport: Transport | None = None,
    root: Path | None = None,
    settings: CollectionSettings | None = None,
) -> TrainingExample:
    """Fetch one non-PDF job into a TrainingExample. Cleans staging before and after."""
    if job.channel == "pdf":
        raise EphemeralError("PDF jobs must go through the ephemeral PDF cursor")
    collection = settings if settings is not None else load_project().collection
    try:
        item = SourceItem(
            source_id=job.source_id,
            source_title=job.source_id,
            title=job.title,
            domain=job.model_domain,
            model_domain=job.model_domain,
            url=job.url,
            license=job.license,
            covers=job.covers,
            kind=job.kind,
        )
        result = fetch_item(item, transport)
        encoded = result.text.encode("utf-8")
        if len(encoded) > collection.max_file_bytes:
            raise EphemeralError(f"{job.title} exceeded max_file_bytes")
        if collection.ephemeral_corpus:
            # Keep text in memory only. Do not wipe .ephemeral/pdf/ mid-book.
            record = {
                "domain": job.model_domain,
                "text": result.text,
                "region": "unspecified",
                "units": "unspecified",
                "license": job.license,
                "source_url": job.url,
                "title": job.title,
                "channel": job.channel,
            }
            example = example_from_record(record, domains)
            if example is None:
                raise EphemeralError(f"{job.title} did not become a training example")
            return example
        staging = ephemeral_directory(root)
        staging.mkdir(parents=True, exist_ok=True)
        path = staging / f"{job.model_domain}.jsonl"
        path.write_text(
            _jsonl_line(job, result.text) + "\n",
            encoding="utf-8",
        )
        example = example_from_record(
            {
                "domain": job.model_domain,
                "text": result.text,
                "region": "unspecified",
                "units": "unspecified",
            },
            domains,
        )
        if example is None:
            raise EphemeralError(f"{job.title} did not become a training example")
        return example
    except (FetchError, SourcesError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise EphemeralError(str(exc)) from exc


class StepPuller:
    """Callable puller that keeps a resumable ephemeral cursor."""

    def __init__(self, cursor: "EphemeralCursor") -> None:
        self.cursor = cursor

    def __call__(self, step: int) -> TrainingExample | None:
        del step
        try:
            return self.cursor.next_example()
        except EphemeralError:
            return None


def make_step_puller(
    jobs: Sequence[EphemeralJob],
    domains: Sequence[str],
    *,
    items_per_step: int,
    transport: Transport | None = None,
    root: Path | None = None,
    settings: CollectionSettings | None = None,
) -> StepPuller | Callable[[int], TrainingExample | None]:
    if items_per_step < 1:
        raise EphemeralError("ephemeral_items_per_step must be positive")
    if not jobs:
        return lambda _step: None

    collection = settings if settings is not None else load_project().collection
    cursor = EphemeralCursor(
        jobs=tuple(jobs),
        domains=tuple(domains),
        transport=transport,
        root=root,
        settings=collection,
    )
    return StepPuller(cursor)


class EphemeralCursor:
    """Walk jobs in order. PDF jobs yield one page per call until the book ends."""

    def __init__(
        self,
        *,
        jobs: tuple[EphemeralJob, ...],
        domains: tuple[str, ...],
        transport: Transport | None,
        root: Path | None,
        settings: CollectionSettings,
    ) -> None:
        self._jobs = jobs
        self._domains = domains
        self._transport = transport
        self._root = root
        self._settings = settings
        self._index = 0
        self._page = 0
        self._pdf = PdfSession(
            root=root,
            max_bytes=settings.pdf_max_bytes,
            max_chars=settings.pdf_chars_per_page,
            transport=transport,
        )

    @property
    def job_index(self) -> int:
        return self._index

    @property
    def page_index(self) -> int:
        return self._page

    @property
    def jobs(self) -> tuple[EphemeralJob, ...]:
        return self._jobs

    def current_job(self) -> EphemeralJob | None:
        if not self._jobs:
            return None
        return self._jobs[self._index % len(self._jobs)]

    def snapshot(self) -> tuple[int, int, str, str]:
        job = self.current_job()
        if job is None:
            return self._index, self._page, "", ""
        return self._index, self._page, job.url, job.title

    def restore(self, job_index: int, page_index: int, job_url: str = "") -> None:
        if not self._jobs:
            return
        if job_index < 0 or page_index < 0:
            raise EphemeralError("cannot restore a negative stage index")
        self._pdf.close()
        self._index = job_index % len(self._jobs)
        self._page = page_index
        if job_url:
            current = self.current_job()
            if current is not None and current.url != job_url:
                # Catalog shifted; find the URL or keep modular index.
                for offset, job in enumerate(self._jobs):
                    if job.url == job_url:
                        self._index = offset
                        break

    def next_example(self) -> TrainingExample | None:
        if not self._jobs:
            return None
        attempts = 0
        while attempts < len(self._jobs):
            job = self._jobs[self._index % len(self._jobs)]
            if job.channel == "pdf":
                try:
                    page, finished = self._pdf.pull_page(job.url, self._page)
                except PdfError as exc:
                    self._pdf.close()
                    self._index = (self._index + 1) % len(self._jobs)
                    self._page = 0
                    attempts += 1
                    if attempts >= len(self._jobs):
                        raise EphemeralError(str(exc)) from exc
                    continue
                record = {
                    "domain": job.model_domain,
                    "text": page.text,
                    "region": "unspecified",
                    "units": "unspecified",
                    "license": job.license,
                    "source_url": page.source_url or job.url,
                    "title": f"{job.title} p{page.page_index + 1}/{page.page_count}",
                    "channel": job.channel,
                }
                example = example_from_record(record, self._domains)
                if example is None:
                    self._page += 1
                    if finished:
                        self._index = (self._index + 1) % len(self._jobs)
                        self._page = 0
                    attempts += 1
                    continue
                if finished:
                    self._index = (self._index + 1) % len(self._jobs)
                    self._page = 0
                else:
                    self._page += 1
                return example

            try:
                example = pull_training_example(
                    job,
                    self._domains,
                    transport=self._transport,
                    root=self._root,
                    settings=self._settings,
                )
            except EphemeralError:
                self._index = (self._index + 1) % len(self._jobs)
                self._page = 0
                attempts += 1
                continue
            self._index = (self._index + 1) % len(self._jobs)
            self._page = 0
            return example
        return None

    def close(self) -> None:
        self._pdf.close()


def _jsonl_line(job: EphemeralJob, text: str) -> str:
    line = json.dumps(
        {
            "domain": job.model_domain,
            "text": text,
            "region": "unspecified",
            "units": "unspecified",
            "license": job.license,
            "source_url": job.url,
            "title": job.title,
            "channel": job.channel,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if "\n" in line or "\r" in line:
        raise EphemeralError("JSONL record contained a raw newline")
    return line
