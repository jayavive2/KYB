"""Evidence store: every Tavily result gets a stable source ID.

IDs are assigned in plan order, so the same recording always produces the same IDs.
That is what lets us compare models citation by citation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Upper bound on text shown to the model per source per category. Tavily's
# `chunks_per_source=3` with `advanced` depth is ~2,400 characters, so this rarely
# truncates; it guards against unusually long chunks.
MAX_CHARS_PER_SOURCE = 2500


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = [(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith("utm_")]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


@dataclass
class Source:
    id: str
    url: str
    title: str
    published_date: str | None
    # Tavily returns query-specific passages, so a page found by two categories can
    # carry different text for each. Keep them apart.
    content_by_category: dict[str, str] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)

    @property
    def categories(self) -> list[str]:
        return list(self.content_by_category)

    def text(self, category: str | None = None) -> str:
        if category is not None:
            return self.content_by_category.get(category, "")
        return "\n...\n".join(dict.fromkeys(self.content_by_category.values()))


class EvidenceStore:
    def __init__(self) -> None:
        self._by_url: dict[str, Source] = {}
        self._by_id: dict[str, Source] = {}

    def add_response(self, category: str, response: dict) -> list[Source]:
        added = []
        query = response.get("query", "")
        for result in response.get("results", []) or []:
            url = result.get("url")
            if not url:
                continue
            key = normalize_url(url)
            source = self._by_url.get(key)
            if source is None:
                source = Source(
                    id=f"S{len(self._by_url) + 1}",
                    url=url,
                    title=(result.get("title") or "").strip() or url,
                    published_date=result.get("published_date"),
                )
                self._by_url[key] = source
                self._by_id[source.id] = source
            content = " ".join((result.get("content") or "").split())
            existing = source.content_by_category.get(category)
            if existing is None:
                source.content_by_category[category] = content
            elif content and content not in existing:
                source.content_by_category[category] = f"{existing}\n...\n{content}"
            if query and query not in source.queries:
                source.queries.append(query)
            if not source.published_date and result.get("published_date"):
                source.published_date = result["published_date"]
            added.append(source)
        return added

    def get(self, source_id: str) -> Source | None:
        return self._by_id.get(source_id)

    def all(self) -> list[Source]:
        return list(self._by_id.values())

    def for_category(self, category: str) -> list[Source]:
        return [s for s in self._by_id.values() if category in s.content_by_category]

    def __len__(self) -> int:
        return len(self._by_id)


def render_sources(sources: list[Source], category: str | None = None,
                   max_chars: int = MAX_CHARS_PER_SOURCE) -> str:
    """Model-facing evidence block: one compact entry per source, labeled with its ID."""
    if not sources:
        return "(no search results)"
    blocks = []
    for s in sources:
        text = s.text(category)
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + " [truncated]"
        blocks.append(
            f"[{s.id}] {s.title}\nURL: {s.url}\nPublished: {s.published_date or 'unknown'}\n{text}"
        )
    return "\n\n".join(blocks)
