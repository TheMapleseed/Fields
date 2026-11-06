"""Ephemeral PDF intake for training.

Large textbooks (OpenStax WEB PDFs are often 100MB+) cannot stay on disk.
The reliable loop is:

1. Resolve a concrete HTTPS PDF URL (OpenStax CMS for /details/books/… pages).
2. Download that one file into data/corpus/.ephemeral/pdf/ (only one book cached).
3. Extract a single page of text for the training step.
4. When the book is finished or training ends, delete the file.

HTTP Range requests are not used: most PDFs need the xref table at the end,
so partial downloads are unreliable for page extraction.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from farm_slm.config import project_root

Transport = Callable[[str, dict[str, str]], bytes]

_USER_AGENT = "FieldsFarmSLM/0.1 (https://github.com/TheMapleseed/Fields; pdf-ephemeral)"
_SLUG = re.compile(r"/details/books/([A-Za-z0-9._-]+)/?$")
_WHITESPACE = re.compile(r"\s+")


class PdfError(RuntimeError):
    pass


@dataclass(frozen=True)
class PdfPageText:
    text: str
    page_index: int
    page_count: int
    source_url: str


def pdf_cache_directory(root: Path | None = None) -> Path:
    base = project_root() if root is None else root
    return base / "data" / "corpus" / ".ephemeral" / "pdf"


def is_openstax_book_page(url: str) -> bool:
    host = urllib.parse.urlparse(url).hostname or ""
    return host.endswith("openstax.org") and "/details/books/" in url


def is_direct_pdf_url(url: str) -> bool:
    path = urllib.parse.urlparse(url).path.lower()
    return path.endswith(".pdf")


def openstax_slug(url: str) -> str | None:
    match = _SLUG.search(urllib.parse.urlparse(url).path)
    return match.group(1) if match else None


def resolve_pdf_url(url: str, transport: Transport | None = None) -> str:
    """Turn an OpenStax details page or a direct PDF link into an https PDF URL."""
    if is_direct_pdf_url(url):
        _require_https(url)
        return url
    slug = openstax_slug(url)
    if slug is None:
        raise PdfError("not an OpenStax book page or direct PDF URL")
    api = (
        "https://openstax.org/apps/cms/api/v2/pages/"
        f"?slug={urllib.parse.quote(slug)}&type=books.Book"
        "&fields=pdf_url,high_resolution_pdf_url,title&format=json"
    )
    payload = json.loads(_http_get(api, transport).decode("utf-8"))
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise PdfError(f"OpenStax CMS has no book for slug {slug}")
    book = items[0]
    if not isinstance(book, dict):
        raise PdfError("OpenStax CMS returned an unexpected book payload")
    pdf = book.get("pdf_url") or book.get("high_resolution_pdf_url")
    if not isinstance(pdf, str) or not pdf.startswith("https://"):
        raise PdfError(f"OpenStax book {slug} has no https pdf_url")
    return pdf


def extract_page_text(data: bytes, page_index: int, *, max_chars: int) -> PdfPageText:
    if page_index < 0:
        raise PdfError("page_index cannot be negative")
    if max_chars < 64:
        raise PdfError("max_chars must be at least 64")
    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise PdfError(f"cannot read PDF: {exc}") from exc
    count = len(reader.pages)
    if count == 0:
        raise PdfError("PDF has no pages")
    if page_index >= count:
        raise PdfError(f"page {page_index} is outside 0..{count - 1}")
    raw = reader.pages[page_index].extract_text() or ""
    text = _WHITESPACE.sub(" ", raw).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    if len(text) < 20:
        raise PdfError(f"page {page_index} has too little extractable text")
    return PdfPageText(text=text, page_index=page_index, page_count=count, source_url="")


def extract_page_from_path(path: Path, page_index: int, *, max_chars: int) -> PdfPageText:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PdfError(f"cannot read {path}") from exc
    return extract_page_text(data, page_index, max_chars=max_chars)


class PdfSession:
    """Cache at most one PDF under .ephemeral/pdf and walk its pages."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        max_bytes: int,
        max_chars: int,
        transport: Transport | None = None,
    ) -> None:
        if max_bytes < 1_000_000:
            raise PdfError("pdf_max_bytes must be at least 1000000")
        self._root = project_root() if root is None else root
        self._max_bytes = max_bytes
        self._max_chars = max_chars
        self._transport = transport
        self._catalog_url: str | None = None
        self._pdf_url: str | None = None
        self._path: Path | None = None
        self._reader: PdfReader | None = None
        self._page_count = 0

    @property
    def active_path(self) -> Path | None:
        return self._path

    def close(self) -> None:
        path = self._path
        self._reader = None
        self._catalog_url = None
        self._pdf_url = None
        self._path = None
        self._page_count = 0
        if path is not None and path.exists():
            path.unlink(missing_ok=True)
        cache = pdf_cache_directory(self._root)
        if cache.is_dir() and not any(cache.iterdir()):
            cache.rmdir()

    def pull_page(self, catalog_url: str, page_index: int) -> tuple[PdfPageText, bool]:
        """Return (page_text, book_finished)."""
        if (
            self._catalog_url != catalog_url
            or self._path is None
            or not self._path.exists()
            or self._reader is None
        ):
            self.close()
            self._open(catalog_url)
        assert self._reader is not None and self._pdf_url is not None
        if page_index < 0 or page_index >= self._page_count:
            raise PdfError(f"page {page_index} is outside 0..{self._page_count - 1}")
        raw = self._reader.pages[page_index].extract_text() or ""
        text = _WHITESPACE.sub(" ", raw).strip()
        if len(text) > self._max_chars:
            text = text[: self._max_chars].rstrip()
        if len(text) < 20:
            raise PdfError(f"page {page_index} has too little extractable text")
        page = PdfPageText(
            text=text,
            page_index=page_index,
            page_count=self._page_count,
            source_url=self._pdf_url,
        )
        finished = page_index + 1 >= self._page_count
        if finished:
            self.close()
        return page, finished

    def _open(self, catalog_url: str) -> None:
        pdf_url = resolve_pdf_url(catalog_url, self._transport)
        raw = _http_get(pdf_url, self._transport)
        if len(raw) > self._max_bytes:
            raise PdfError(
                f"PDF is {len(raw)} bytes; pdf_max_bytes is {self._max_bytes}"
            )
        cache = pdf_cache_directory(self._root)
        cache.mkdir(parents=True, exist_ok=True)
        # Only one cached book: clear siblings first.
        for existing in cache.glob("*.pdf"):
            existing.unlink(missing_ok=True)
        digest = hashlib.sha256(pdf_url.encode("utf-8")).hexdigest()[:16]
        path = cache / f"{digest}.pdf"
        path.write_bytes(raw)
        try:
            reader = PdfReader(str(path))
            page_count = len(reader.pages)
        except PdfReadError as exc:
            path.unlink(missing_ok=True)
            raise PdfError(f"cannot read PDF: {exc}") from exc
        if page_count < 1:
            path.unlink(missing_ok=True)
            raise PdfError("PDF has no pages")
        self._catalog_url = catalog_url
        self._pdf_url = pdf_url
        self._path = path
        self._reader = reader
        self._page_count = page_count


def _require_https(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise PdfError("refusing a non-https PDF URL")


def _http_get(url: str, transport: Transport | None) -> bytes:
    _require_https(url)
    headers = {"User-Agent": _USER_AGENT, "Accept": "*/*"}
    if transport is not None:
        return transport(url, headers)
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            final = urllib.parse.urlparse(response.geturl())
            if final.scheme != "https":
                raise PdfError("refusing a non-https redirect")
            return response.read()
    except urllib.error.URLError as exc:
        raise PdfError(f"download failed: {exc}") from exc
