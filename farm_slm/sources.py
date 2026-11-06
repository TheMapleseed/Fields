"""Read sources.kdl and expose collection targets. This module does not download."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from farm_slm.config import load_project, project_root
from farm_slm.corpus import CorpusError, resolve_domain
from farm_slm.kdl import KdlError, Node, parse_kdl


class SourcesError(ValueError):
    pass


@dataclass(frozen=True)
class SourceItem:
    source_id: str
    source_title: str
    title: str
    domain: str
    model_domain: str
    url: str
    license: str
    covers: str
    kind: str


@dataclass(frozen=True)
class SourceCatalog:
    items: tuple[SourceItem, ...]

    def by_model_domain(self) -> dict[str, tuple[SourceItem, ...]]:
        grouped: dict[str, list[SourceItem]] = {}
        for item in self.items:
            grouped.setdefault(item.model_domain, []).append(item)
        return {key: tuple(value) for key, value in grouped.items()}


@dataclass(frozen=True)
class DeferredTarget:
    """A catalog item that belongs to maps or dump intake, not the text fetchers."""

    title: str
    source_id: str
    kind: str
    channel: str
    url: str
    license: str
    covers: str
    model_domain: str
    intake: str


def load_source_catalog(path: Path | None = None) -> SourceCatalog:
    catalog_path = project_root() / "sources.kdl" if path is None else path
    try:
        text = catalog_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SourcesError(f"cannot read {catalog_path}") from exc
    try:
        nodes = parse_kdl(text)
    except KdlError as exc:
        raise SourcesError(f"invalid sources catalog: {exc}") from exc
    allowed = load_project().model.domains
    items: list[SourceItem] = []
    for node in nodes:
        if node.name != "source":
            continue
        if len(node.args) != 1 or not isinstance(node.args[0], str) or not node.args[0]:
            raise SourcesError("each source needs one name")
        source_id = node.args[0]
        source_title = _optional_string(node, "title") or source_id
        kind = _optional_string(node, "kind") or "unknown"
        for child in node.children:
            if child.name != "item":
                continue
            items.append(_item(source_id, source_title, kind, child, allowed))
    if not items:
        raise SourcesError("sources.kdl lists no items")
    return SourceCatalog(tuple(items))


def github_plan_from_catalog(catalog: SourceCatalog | None = None) -> dict[str, list[str]]:
    """Build GitHub search slots from catalog titles. Does not call GitHub."""
    project = load_project()
    limit = project.collection.max_queries_per_domain
    current = catalog if catalog is not None else load_source_catalog()
    plan = {domain: [] for domain in project.model.domains}
    seen: dict[str, set[str]] = {domain: set() for domain in project.model.domains}
    for item in current.items:
        bucket = plan[item.model_domain]
        if len(bucket) >= limit:
            continue
        query = _github_query(item)
        if query in seen[item.model_domain]:
            continue
        seen[item.model_domain].add(query)
        bucket.append(query)
    return plan


def map_targets_from_catalog(catalog: SourceCatalog | None = None) -> tuple[DeferredTarget, ...]:
    """Geospatial catalog items for local GeoTIFF intake through farm_slm.maps."""
    return _deferred_targets(catalog, channels={"deferred_geospatial"}, intake="maps")


def dump_targets_from_catalog(catalog: SourceCatalog | None = None) -> tuple[DeferredTarget, ...]:
    """Bulk dump and binary document items that need dedicated readers."""
    return _deferred_targets(
        catalog,
        channels={"deferred_dump", "deferred_document"},
        intake="dump",
    )


def deferred_pipeline_report(catalog: SourceCatalog | None = None) -> str:
    current = catalog if catalog is not None else load_source_catalog()
    maps = map_targets_from_catalog(current)
    dumps = dump_targets_from_catalog(current)
    lines = [
        "Deferred pipelines",
        f"maps_targets={len(maps)} (place GeoTIFFs under data/maps/sheets/)",
        f"dump_targets={len(dumps)} (dedicated dump/document readers)",
    ]
    for target in maps:
        lines.append(
            f"  maps/{target.kind}: {target.title} [{target.license}] {target.url}"
        )
    for target in dumps:
        lines.append(
            f"  {target.channel}: {target.title} [{target.license}] {target.url}"
        )
    lines.append(
        "Maps intake uses write_coordinate_example() on a local sheet; dumps stay unread until a reader lands."
    )
    return "\n".join(lines)


def collection_targets(catalog: SourceCatalog | None = None) -> str:
    from farm_slm.fetch import primary_plan, routing_report

    current = catalog if catalog is not None else load_source_catalog()
    fetch_on = load_project().collection.fetch_catalog
    lines = [
        "Source catalog targets",
        f"items={len(current.items)}",
        f"fetch={'on' if fetch_on else 'off'}",
    ]
    for domain, items in sorted(current.by_model_domain().items()):
        lines.append(f"{domain}: {len(items)} items")
        for item in items[:3]:
            channel = primary_plan(item).channel
            lines.append(f"  - {item.title} [{item.license}] ({channel}) {item.url}")
        if len(items) > 3:
            lines.append(f"  - … {len(items) - 3} more")
    lines.append(routing_report(current.items))
    lines.append(deferred_pipeline_report(current))
    lines.append(
        "Permanent collect stays off. Training uses fetch_during_train + ephemeral_corpus."
    )
    lines.append("Call collect_github(catalog_plan()) only if a machine can keep GitHub text.")
    return "\n".join(lines)


def _deferred_targets(
    catalog: SourceCatalog | None,
    *,
    channels: set[str],
    intake: str,
) -> tuple[DeferredTarget, ...]:
    from farm_slm.fetch import primary_plan

    current = catalog if catalog is not None else load_source_catalog()
    targets: list[DeferredTarget] = []
    for item in current.items:
        plan = primary_plan(item)
        if plan.channel not in channels:
            continue
        targets.append(
            DeferredTarget(
                title=item.title,
                source_id=item.source_id,
                kind=item.kind,
                channel=plan.channel,
                url=item.url,
                license=item.license,
                covers=item.covers,
                model_domain=item.model_domain,
                intake=intake,
            )
        )
    return tuple(targets)


def _item(
    source_id: str,
    source_title: str,
    kind: str,
    node: Node,
    allowed: Sequence[str],
) -> SourceItem:
    title = _required_string(node, "title")
    domain = _required_string(node, "domain")
    url = _required_string(node, "url")
    license_id = _required_string(node, "license")
    covers = _optional_string(node, "covers") or ""
    if not url.startswith("https://"):
        raise SourcesError(f"{title} needs an https url")
    try:
        model_domain = resolve_domain(domain, allowed)
    except CorpusError as exc:
        raise SourcesError(str(exc)) from exc
    return SourceItem(
        source_id=source_id,
        source_title=source_title,
        title=title,
        domain=domain,
        model_domain=model_domain,
        url=url,
        license=license_id,
        covers=covers,
        kind=kind,
    )


def _github_query(item: SourceItem) -> str:
    words = " ".join(part for part in item.title.replace(":", " ").split() if part)[:80]
    return f"{words} agriculture in:readme,description"


def _required_string(node: Node, name: str) -> str:
    value = _optional_string(node, name)
    if value is None:
        raise SourcesError(f"item is missing {name}")
    return value


def _optional_string(node: Node, name: str) -> str | None:
    matches = [child for child in node.children if child.name == name]
    if not matches:
        return None
    if len(matches) != 1 or len(matches[0].args) != 1 or not isinstance(matches[0].args[0], str):
        raise SourcesError(f"{name} must be one string")
    value = matches[0].args[0].strip()
    return value or None


def main() -> None:
    from farm_slm.collect import collect_catalog

    catalog = load_source_catalog()
    print(collection_targets(catalog))
    plan = github_plan_from_catalog(catalog)
    print("GitHub plan slots")
    for domain, queries in plan.items():
        print(f"{domain}: {len(queries)} queries")
    summary = collect_catalog(fetch=False, catalog=catalog)
    print(summary.note)


if __name__ == "__main__":
    main()
