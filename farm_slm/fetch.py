"""Fetch adapters for sources.kdl items.

Every catalog item is routed to a channel. Channels know how to download text
when fetch is on. Binary map and dump targets stay deferred to the maps and
dump pipelines. This module does not run a fetch unless collect_catalog asks.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Protocol

from farm_slm.sources import SourceItem

Transport = Callable[[str, dict[str, str]], bytes]

_USER_AGENT = "FieldsFarmSLM/0.1 (https://github.com/TheMapleseed/Fields; catalog-collector)"
_WHITESPACE = re.compile(r"\s+")


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChannelPlan:
    channel: str
    fetchable: bool
    reason: str


@dataclass(frozen=True)
class FetchResult:
    text: str
    content_type: str


class Fetcher(Protocol):
    name: str

    def plan(self, item: SourceItem) -> ChannelPlan | None:
        ...

    def fetch(self, item: SourceItem, transport: Transport | None = None) -> FetchResult:
        ...


class DictionaryFetcher:
    name = "dictionary"

    def plan(self, item: SourceItem) -> ChannelPlan | None:
        host = urllib.parse.urlparse(item.url).hostname or ""
        if host != "api.dictionaryapi.dev" and item.kind != "english-dictionary":
            return None
        word = _dictionary_word(item.url)
        if word is None:
            return ChannelPlan(
                self.name,
                False,
                "expand english_seeds.kdl against this API during ephemeral train",
            )
        return ChannelPlan(
            self.name,
            True,
            "dictionaryapi.dev English Wiktionary entry (CC BY-SA)",
        )

    def fetch(self, item: SourceItem, transport: Transport | None = None) -> FetchResult:
        word = _dictionary_word(item.url)
        if word is None:
            raise FetchError("dictionary URL must end with a concrete word")
        raw = _http_get(item.url, transport)
        payload = json.loads(raw.decode("utf-8"))
        text = _dictionary_text(payload, word)
        if len(text) < 20:
            raise FetchError(f"dictionary entry too short for {word}")
        return FetchResult(text, "dictionary-entry")


class WikipediaFetcher:
    name = "wikipedia"

    def plan(self, item: SourceItem) -> ChannelPlan | None:
        host = urllib.parse.urlparse(item.url).hostname or ""
        if host.endswith("wikipedia.org") and "/wiki/" in item.url:
            return ChannelPlan(self.name, True, "MediaWiki plain-text extract")
        return None

    def fetch(self, item: SourceItem, transport: Transport | None = None) -> FetchResult:
        title = urllib.parse.unquote(item.url.rsplit("/wiki/", 1)[-1])
        query = urllib.parse.urlencode(
            {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "explaintext": "1",
                "redirects": "1",
                "titles": title.replace("_", " "),
            }
        )
        api = f"https://en.wikipedia.org/w/api.php?{query}"
        payload = json.loads(_http_get(api, transport).decode("utf-8"))
        pages = payload.get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract")
            if isinstance(extract, str) and extract.strip():
                return FetchResult(extract.strip(), "wikipedia-extract")
        raise FetchError(f"no Wikipedia extract for {item.title}")


class HtmlFetcher:
    name = "html"

    def plan(self, item: SourceItem) -> ChannelPlan | None:
        from farm_slm.pdf import is_openstax_book_page

        host = urllib.parse.urlparse(item.url).hostname or ""
        if host.endswith("wikipedia.org"):
            return None
        if item.kind in _DEFERRED_KINDS:
            return None
        if item.kind == "textbooks" and is_openstax_book_page(item.url):
            return None
        if any(token in item.url for token in _DUMP_MARKERS):
            return None
        if _binary_document_url(item.url):
            return None
        return ChannelPlan(self.name, True, "HTTPS page as plain text")

    def fetch(self, item: SourceItem, transport: Transport | None = None) -> FetchResult:
        raw = _http_get(item.url, transport)
        text = _html_to_text(raw.decode("utf-8", errors="replace"))
        if len(text) < 40:
            raise FetchError(f"page text too short for {item.title}")
        return FetchResult(text, "html-text")


class JsonDocumentFetcher:
    name = "json_document"

    def plan(self, item: SourceItem) -> ChannelPlan | None:
        if item.kind in _DEFERRED_KINDS:
            return None
        if "geojson" in item.url or item.url.endswith(".json") or "format=json" in item.url:
            return ChannelPlan(self.name, True, "HTTPS JSON document")
        return None

    def fetch(self, item: SourceItem, transport: Transport | None = None) -> FetchResult:
        raw = _http_get(item.url, transport)
        payload = json.loads(raw.decode("utf-8"))
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return FetchResult(text, "json-document")


class DeferredBinaryFetcher:
    name = "deferred_geospatial"

    def plan(self, item: SourceItem) -> ChannelPlan | None:
        if item.kind in _DEFERRED_KINDS:
            return ChannelPlan(
                self.name,
                False,
                "georeferenced sheets and imagery go through farm_slm.maps, not the text corpus",
            )
        return None

    def fetch(self, item: SourceItem, transport: Transport | None = None) -> FetchResult:
        raise FetchError(f"{item.title} is deferred to the maps pipeline")


class DeferredDumpFetcher:
    name = "deferred_dump"

    def plan(self, item: SourceItem) -> ChannelPlan | None:
        if any(token in item.url for token in _DUMP_MARKERS):
            return ChannelPlan(
                self.name,
                False,
                "bulk dumps need a dedicated dump reader; catalog entry is wired only",
            )
        return None

    def fetch(self, item: SourceItem, transport: Transport | None = None) -> FetchResult:
        raise FetchError(f"{item.title} is a bulk dump and is not auto-fetched")


class PdfDocumentFetcher:
    """OpenStax books and direct PDF URLs. Text comes from PdfSession page walks."""

    name = "pdf"

    def plan(self, item: SourceItem) -> ChannelPlan | None:
        from farm_slm.pdf import is_direct_pdf_url, is_openstax_book_page

        if is_direct_pdf_url(item.url) or (
            item.kind == "textbooks" and is_openstax_book_page(item.url)
        ):
            return ChannelPlan(
                self.name,
                False,
                "download one PDF into .ephemeral, extract a page per train step, then delete",
            )
        return None

    def fetch(self, item: SourceItem, transport: Transport | None = None) -> FetchResult:
        raise FetchError("PDFs are read page-by-page during ephemeral training")


class DeferredDocumentFetcher:
    name = "deferred_document"

    def plan(self, item: SourceItem) -> ChannelPlan | None:
        if _binary_document_url(item.url) and not item.url.lower().endswith(".pdf"):
            return ChannelPlan(
                self.name,
                False,
                "binary documents need a dedicated reader; catalog entry is wired only",
            )
        return None

    def fetch(self, item: SourceItem, transport: Transport | None = None) -> FetchResult:
        raise FetchError(f"{item.title} is a binary document and is not auto-fetched")


class GithubSlotFetcher:
    """Marks items that also seed GitHub search slots. Fetch is collect_github."""

    name = "github_slot"

    def plan(self, item: SourceItem) -> ChannelPlan:
        return ChannelPlan(
            self.name,
            False,
            "also available as a GitHub search query via catalog_plan()",
        )

    def fetch(self, item: SourceItem, transport: Transport | None = None) -> FetchResult:
        raise FetchError("GitHub text is collected with collect_github(catalog_plan())")


_DEFERRED_KINDS = {
    "elevation",
    "imagery",
    "foliage",
    "topography",
    "map",
}
_DUMP_MARKERS = (
    "dumps.wikimedia.org",
    "planet.openstreetmap.org",
    "registry.opendata.aws",
    "opendata.aws",
)
_BINARY_SUFFIXES = (
    ".tif",
    ".tiff",
    ".zip",
    ".tar",
    ".gz",
    ".nc",
    ".pdf",
    ".hdf",
    ".h5",
    ".gpkg",
    ".shp",
)

# Order matters: first matching specialized plan wins. github_slot is additive.
_FETCHERS: tuple[Fetcher, ...] = (
    DeferredBinaryFetcher(),
    DeferredDumpFetcher(),
    PdfDocumentFetcher(),
    DeferredDocumentFetcher(),
    DictionaryFetcher(),
    WikipediaFetcher(),
    JsonDocumentFetcher(),
    HtmlFetcher(),
)


def route_item(item: SourceItem) -> tuple[ChannelPlan, ...]:
    plans: list[ChannelPlan] = []
    for fetcher in _FETCHERS:
        plan = fetcher.plan(item)
        if plan is not None:
            plans.append(plan)
            break
    else:
        plans.append(ChannelPlan("unrouted", False, "no fetcher matched this item"))
    plans.append(GithubSlotFetcher().plan(item))
    return tuple(plans)


def primary_plan(item: SourceItem) -> ChannelPlan:
    return route_item(item)[0]


def fetch_item(item: SourceItem, transport: Transport | None = None) -> FetchResult:
    plan = primary_plan(item)
    if not plan.fetchable:
        raise FetchError(plan.reason)
    for fetcher in _FETCHERS:
        matched = fetcher.plan(item)
        if matched is not None and matched.channel == plan.channel:
            return fetcher.fetch(item, transport)
    raise FetchError(f"no fetcher for channel {plan.channel}")


def routing_report(items: tuple[SourceItem, ...]) -> str:
    counts: dict[str, int] = {}
    fetchable = 0
    for item in items:
        plan = primary_plan(item)
        counts[plan.channel] = counts.get(plan.channel, 0) + 1
        if plan.fetchable:
            fetchable += 1
    lines = [
        "Catalog fetch routing",
        f"items={len(items)}",
        f"fetchable={fetchable}",
        f"deferred={len(items) - fetchable}",
    ]
    for channel, count in sorted(counts.items()):
        lines.append(f"{channel}: {count}")
    return "\n".join(lines)


def _binary_document_url(url: str) -> bool:
    path = urllib.parse.urlparse(url).path.lower()
    return any(path.endswith(suffix) for suffix in _BINARY_SUFFIXES)


def _http_get(url: str, transport: Transport | None) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise FetchError("refusing a non-https URL")
    headers = {"User-Agent": _USER_AGENT, "Accept": "*/*"}
    if transport is not None:
        return transport(url, headers)
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            final = urllib.parse.urlparse(response.geturl())
            if final.scheme != "https":
                raise FetchError("refusing a non-https redirect")
            return response.read()
    except urllib.error.URLError as exc:
        raise FetchError(f"download failed: {exc}") from exc


def _dictionary_word(url: str) -> str | None:
    path = urllib.parse.urlparse(url).path.rstrip("/")
    word = urllib.parse.unquote(path.rsplit("/", 1)[-1])
    if not word or word in {"en", "entries", "v2", "api"}:
        return None
    if not re.fullmatch(r"[A-Za-z][A-Za-z'-]{0,63}", word):
        return None
    return word


def _dictionary_text(payload: object, word: str) -> str:
    if not isinstance(payload, list) or not payload:
        raise FetchError(f"no dictionary senses for {word}")
    parts: list[str] = [f"Word: {word}"]
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        meanings = entry.get("meanings")
        if not isinstance(meanings, list):
            continue
        for meaning in meanings:
            if not isinstance(meaning, dict):
                continue
            pos = meaning.get("partOfSpeech", "unknown")
            definitions = meaning.get("definitions")
            if not isinstance(definitions, list):
                continue
            for definition in definitions[:3]:
                if not isinstance(definition, dict):
                    continue
                gloss = definition.get("definition")
                if not isinstance(gloss, str) or not gloss.strip():
                    continue
                line = f"{pos}: {gloss.strip()}"
                example = definition.get("example")
                if isinstance(example, str) and example.strip():
                    line += f" Example: {example.strip()}"
                parts.append(line)
    return "\n".join(parts).strip()


def _html_to_text(document: str) -> str:
    parser = _TextExtractor()
    parser.feed(document)
    parser.close()
    return _WHITESPACE.sub(" ", parser.text).strip()


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip = 0

    @property
    def text(self) -> str:
        return " ".join(self._chunks)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        piece = data.strip()
        if piece:
            self._chunks.append(piece)
