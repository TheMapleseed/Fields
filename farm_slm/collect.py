"""GitHub collection slots for the farming language model.

Queries start empty. Add them to the plan, then call collect_github. The
collector stores text. It does not clone repositories or run their code.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from farm_slm.config import CollectionSettings, load_project, project_root

DOMAINS: tuple[str, ...] = load_project().model.domains

_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_BRANCH = re.compile(r"^[A-Za-z0-9._/-]+$")
_TEXT_EXTENSIONS = {".md", ".markdown", ".rst", ".txt", ".ipynb"}
_SKIP_DIRS = {"node_modules", "vendor", "dist", "__pycache__", ".git"}
_SECRET_NAMES = {
    ".env",
    ".npmrc",
    ".pypirc",
    "id_rsa",
    "id_dsa",
    "credentials.json",
    "secrets.json",
}
_SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".kdbx"}
_SECRET_TEXT = re.compile(
    r"(-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"AKIA[0-9A-Z]{16}|"
    r"github_pat_[A-Za-z0-9_]{20,}|"
    r"ghp_[A-Za-z0-9]{20,})"
)
_API = "https://api.github.com"

Transport = Callable[[str, dict[str, str]], dict[str, object]]


class CollectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CollectionSummary:
    written: int
    files: tuple[str, ...]
    skipped: tuple[str, ...]
    note: str


def empty_plan() -> dict[str, list[str]]:
    return {domain: [] for domain in DOMAINS}


def collection_section(plan: Mapping[str, Sequence[str]] | None = None) -> str:
    current = empty_plan() if plan is None else plan
    lines = ["GitHub collection slots"]
    for domain in DOMAINS:
        queries = list(current.get(domain, []))
        if queries:
            rendered = "; ".join(queries)
        else:
            rendered = "(no queries yet)"
        lines.append(f"{domain}: {rendered}")
    lines.append(
        "record fields: domain, text, region, units, repo, path, sha, license, source_url"
    )
    lines.append("Add queries to the plan, then call collect_github(plan).")
    return "\n".join(lines)


def collect_github(
    plan: Mapping[str, Sequence[str]],
    root: Path | None = None,
    token: str | None = None,
    transport: Transport | None = None,
) -> CollectionSummary:
    project = load_project()
    settings = project.collection
    _validate_plan(plan, settings)
    if all(len(tuple(plan[domain])) == 0 for domain in DOMAINS):
        return CollectionSummary(0, (), (), "No GitHub queries yet.")

    base = project_root() if root is None else root
    client = GitHubClient(token=token, transport=transport)
    written = 0
    files: list[str] = []
    skipped: list[str] = []
    total_bytes = 0
    for domain in DOMAINS:
        queries = tuple(plan[domain])
        if not queries:
            continue
        records: list[dict[str, str]] = []
        seen: set[str] = set()
        for query in queries:
            try:
                repos = client.search_repositories(query, settings.max_repos_per_query)
            except CollectionError as exc:
                skipped.append(f"{domain}: {exc}")
                continue
            for repo in repos:
                added, used, notes = _collect_repo(
                    client,
                    repo,
                    domain,
                    settings,
                    seen,
                    total_bytes,
                )
                total_bytes += used
                records.extend(added)
                skipped.extend(notes)
                if total_bytes >= settings.max_total_bytes:
                    break
            if total_bytes >= settings.max_total_bytes:
                skipped.append("stopped at max_total_bytes")
                break
        if records:
            relative = _write_domain(base, domain, records)
            files.append(relative)
            written += len(records)
    note = "Collected GitHub text." if written else "No text files matched the queries."
    return CollectionSummary(written, tuple(files), tuple(skipped), note)


def select_text_blobs(
    tree: Sequence[Mapping[str, object]],
    max_file_bytes: int,
    max_files: int,
) -> list[tuple[str, str, int]]:
    selected: list[tuple[str, str, int]] = []
    for entry in tree:
        if entry.get("type") != "blob":
            continue
        path = entry.get("path")
        sha = entry.get("sha")
        size = entry.get("size", 0)
        if not isinstance(path, str) or not isinstance(sha, str):
            continue
        if not isinstance(size, int) or isinstance(size, bool):
            continue
        if not _allowed_path(path, size, max_file_bytes):
            continue
        if not _SHA.fullmatch(sha):
            continue
        selected.append((path, sha, size))
        if len(selected) >= max_files:
            break
    return selected


class GitHubClient:
    def __init__(self, token: str | None = None, transport: Transport | None = None) -> None:
        if token is None:
            token = os.environ.get("GITHUB_TOKEN") or None
        if token == "":
            token = None
        if token is not None and (len(token) > 200 or any(char.isspace() for char in token)):
            raise CollectionError("GITHUB_TOKEN is malformed")
        self._token = token
        self._transport = transport

    def search_repositories(self, query: str, limit: int) -> list[dict[str, object]]:
        params = urllib.parse.urlencode(
            {
                "q": query,
                "per_page": str(min(limit, 100)),
                "sort": "stars",
                "order": "desc",
            }
        )
        payload = self._get(f"/search/repositories?{params}")
        items = payload.get("items")
        if not isinstance(items, list):
            raise CollectionError("GitHub search returned an unexpected payload")
        repos: list[dict[str, object]] = []
        for item in items:
            if isinstance(item, dict):
                repos.append(item)
            if len(repos) >= limit:
                break
        return repos

    def repository_tree(self, full_name: str, branch: str) -> dict[str, object]:
        owner, repo = _split_repo(full_name)
        if not _BRANCH.fullmatch(branch) or ".." in branch.split("/"):
            raise CollectionError("refusing an unexpected git branch name")
        quoted = urllib.parse.quote(branch, safe="")
        return self._get(f"/repos/{owner}/{repo}/git/trees/{quoted}?recursive=1")

    def blob_text(self, full_name: str, sha: str, path: str) -> str | None:
        owner, repo = _split_repo(full_name)
        if not _SHA.fullmatch(sha):
            raise CollectionError("refusing an unexpected blob sha")
        payload = self._get(f"/repos/{owner}/{repo}/git/blobs/{sha}")
        if payload.get("encoding") != "base64":
            return None
        content = payload.get("content")
        if not isinstance(content, str):
            return None
        raw = _decode_base64(content)
        if raw is None:
            return None
        return text_from_bytes(path, raw)

    def _get(self, path: str) -> dict[str, object]:
        url = _github_url(path)
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "farm-slm-collector",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token}"
        if self._transport is not None:
            payload = self._transport(url, headers)
            if not isinstance(payload, dict):
                raise CollectionError("GitHub transport returned an unexpected payload")
            return payload
        request = urllib.request.Request(url, headers=headers)
        opener = urllib.request.build_opener(_GitHubRedirectHandler)
        try:
            with opener.open(request, timeout=30) as response:
                body = response.read(8_000_001)
                if len(body) > 8_000_000:
                    raise CollectionError("GitHub response is too large")
        except urllib.error.HTTPError as exc:
            if exc.code in {403, 429}:
                raise CollectionError(
                    "GitHub rate limit reached. Set GITHUB_TOKEN and try again."
                ) from None
            raise CollectionError(f"GitHub HTTP {exc.code}") from None
        except urllib.error.URLError as exc:
            raise CollectionError("GitHub request failed") from None
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise CollectionError("GitHub returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise CollectionError("GitHub returned an unexpected payload")
        return payload


class _GitHubRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme != "https" or parsed.hostname != "api.github.com":
            raise CollectionError("refused a redirect away from api.github.com")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def text_from_bytes(path: str, raw: bytes) -> str | None:
    if len(raw) == 0 or b"\x00" in raw:
        return None
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if path.lower().endswith(".ipynb"):
        decoded = _notebook_text(decoded)
        if decoded is None:
            return None
    if _SECRET_TEXT.search(decoded):
        return None
    stripped = decoded.strip()
    if not stripped:
        return None
    return stripped


def _notebook_text(raw: str) -> str | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    cells = payload.get("cells")
    if not isinstance(cells, list):
        return None
    parts: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            if not all(isinstance(part, str) for part in source):
                continue
            source = "".join(source)
        if isinstance(source, str) and source.strip():
            parts.append(source.strip())
    if not parts:
        return None
    return "\n\n".join(parts)


def _collect_repo(
    client: GitHubClient,
    repo: Mapping[str, object],
    domain: str,
    settings: CollectionSettings,
    seen: set[str],
    total_bytes: int,
) -> tuple[list[dict[str, str]], int, list[str]]:
    full_name = repo.get("full_name")
    branch = repo.get("default_branch")
    html_url = repo.get("html_url")
    if not isinstance(full_name, str) or not isinstance(branch, str):
        return [], 0, ["skipped a repository with an unexpected name"]
    try:
        _split_repo(full_name)
    except CollectionError:
        return [], 0, [f"skipped {full_name}"]
    try:
        tree_payload = client.repository_tree(full_name, branch)
    except CollectionError as exc:
        return [], 0, [f"{full_name}: {exc}"]
    if tree_payload.get("truncated") is True:
        return [], 0, [f"{full_name}: tree truncated"]
    tree = tree_payload.get("tree")
    if not isinstance(tree, list):
        return [], 0, [f"{full_name}: missing tree"]
    entries = [item for item in tree if isinstance(item, dict)]
    chosen = select_text_blobs(entries, settings.max_file_bytes, settings.max_files_per_repo)
    records: list[dict[str, str]] = []
    used = 0
    notes: list[str] = []
    license_id = _license_id(repo.get("license"))
    source_root = html_url if isinstance(html_url, str) else f"https://github.com/{full_name}"
    for path, sha, size in chosen:
        identity = f"{full_name}@{sha}"
        if identity in seen:
            continue
        if total_bytes + used + size > settings.max_total_bytes:
            notes.append("stopped at max_total_bytes")
            break
        try:
            text = client.blob_text(full_name, sha, path)
        except CollectionError as exc:
            notes.append(f"{full_name}/{path}: {exc}")
            continue
        if text is None:
            continue
        seen.add(identity)
        used += len(text.encode("utf-8"))
        records.append(
            {
                "domain": domain,
                "repo": full_name,
                "path": path,
                "sha": sha,
                "license": license_id,
                "source_url": _source_url(source_root, branch, path),
                "region": "unspecified",
                "units": "unspecified",
                "text": text,
            }
        )
    return records, used, notes


def _write_domain(root: Path, domain: str, records: Sequence[Mapping[str, str]]) -> str:
    if domain not in DOMAINS:
        raise CollectionError("refusing to write an unknown domain")
    relative = Path("data") / "corpus" / f"{domain}.jsonl"
    destination = root / relative
    if destination.resolve().parent != (root / "data" / "corpus").resolve():
        raise CollectionError("refusing to write outside the corpus directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".jsonl.partial")
    lines: list[str] = []
    for record in records:
        line = json.dumps(dict(record), ensure_ascii=False, separators=(",", ":"))
        if "\n" in line or "\r" in line:
            raise CollectionError("JSONL record contained a raw newline")
        lines.append(line)
    try:
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return relative.as_posix()


def _validate_plan(plan: Mapping[str, Sequence[str]], settings: CollectionSettings) -> None:
    if set(plan) != set(DOMAINS):
        raise CollectionError("the plan must contain every domain exactly once")
    for domain in DOMAINS:
        queries = tuple(plan[domain])
        if len(queries) > settings.max_queries_per_domain:
            raise CollectionError(f"{domain} has too many queries")
        for query in queries:
            if not isinstance(query, str) or not query.strip():
                raise CollectionError(f"{domain} has an empty query")
            if len(query) > 256 or any(char in query for char in "\r\n"):
                raise CollectionError(f"{domain} has a query that cannot be sent")


def _allowed_path(path: str, size: int, max_file_bytes: int) -> bool:
    if size <= 0 or size > max_file_bytes or len(path) > 512 or "\\" in path:
        return False
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    if any(part in _SKIP_DIRS for part in parts[:-1]):
        return False
    name = parts[-1].lower()
    if name in _SECRET_NAMES or Path(name).suffix.lower() in _SECRET_SUFFIXES:
        return False
    return Path(name).suffix.lower() in _TEXT_EXTENSIONS


def _split_repo(full_name: str) -> tuple[str, str]:
    owner, separator, repo = full_name.partition("/")
    if separator != "/" or not _SEGMENT.fullmatch(owner) or not _SEGMENT.fullmatch(repo):
        raise CollectionError("refusing an unexpected repository name")
    return owner, repo


def _license_id(value: object) -> str:
    if not isinstance(value, dict):
        return "unknown"
    spdx = value.get("spdx_id")
    if not isinstance(spdx, str) or not spdx or spdx == "NOASSERTION":
        return "unknown"
    if len(spdx) > 64 or any(char.isspace() for char in spdx):
        return "unknown"
    return spdx


def _source_url(html_url: str, branch: str, path: str) -> str:
    parsed = urllib.parse.urlparse(html_url)
    if parsed.scheme != "https" or parsed.hostname != "github.com":
        raise CollectionError("refusing an unexpected repository URL")
    quoted_branch = urllib.parse.quote(branch, safe="")
    quoted_path = urllib.parse.quote(path, safe="/")
    return f"{parsed.scheme}://{parsed.hostname}{parsed.path}/blob/{quoted_branch}/{quoted_path}"


def _decode_base64(content: str) -> bytes | None:
    compact = "".join(content.split())
    if not compact or len(compact) % 4 != 0:
        return None
    if re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", compact) is None:
        return None
    try:
        return base64.b64decode(compact)
    except ValueError:
        return None


def _github_url(path: str) -> str:
    if not path.startswith("/") or path.startswith("//"):
        raise CollectionError("refusing a non-API path")
    url = _API + path
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "api.github.com":
        raise CollectionError("refusing a non-API URL")
    return url
