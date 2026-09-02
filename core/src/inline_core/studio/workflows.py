"""The published workflow catalogue, proxied for the app's Workflows popup.

The API is CORS-open, so the renderer could read it directly; it comes through Core only so the
popup survives a machine with no network. Hero media is deliberately not proxied - copying video
bytes through Core buys no offline behaviour and costs a full extra copy.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from . import config as cfg

#: Overridable so a staging site can be pointed at without a rebuild.
DEFAULT_CATALOGUE_URL = "https://inlinestudio.art"

_TIMEOUT = 10


def catalogue_url() -> str:
    return (os.environ.get("INLINE_STUDIO_WORKFLOWS_URL") or DEFAULT_CATALOGUE_URL).rstrip("/")


def _cache_dir() -> Path:
    return cfg.data_dir() / "workflows-cache"


def install_id() -> str:
    """A stable anonymous id for this install, so app installs count apart from web visitors.

    Generated once, never derived from anything about the user or the machine, and deliberately
    shipped with no opt-out - the privacy policy has to cover it.
    """
    path = cfg.data_dir() / "install-id"
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except OSError:
        pass
    fresh = str(uuid.uuid4())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(fresh, encoding="utf-8")
    except OSError:
        pass  # A read-only data dir still gets a working popup, just an unstable id.
    return fresh


def _headers(app_version: str) -> dict[str, str]:
    return {
        "X-Inline-Client": app_version or "unknown",
        "X-Inline-Install-Id": install_id(),
        "Accept": "application/json",
    }


@dataclass
class _Fetched:
    body: Any
    etag: str | None
    #: True on a 304: the cache is current and must not be overwritten with an empty body.
    unchanged: bool = False


def _fetch(url: str, app_version: str, etag: str | None) -> _Fetched | None:
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    request = Request(url, headers=_headers(app_version))
    if etag:
        request.add_header("If-None-Match", etag)
    try:
        with urlopen(request, timeout=_TIMEOUT) as response:  # noqa: S310 - URL comes from config
            return _Fetched(
                body=json.loads(response.read().decode("utf-8")),
                etag=response.headers.get("ETag"),
            )
    except HTTPError as error:
        if error.code == 304:
            return _Fetched(body=None, etag=etag, unchanged=True)
        return None
    except (URLError, OSError, ValueError):
        return None


def _cache_path(key: str) -> Path:
    return _cache_dir() / f"{key}.json"


def _read_cache(key: str) -> tuple[Any | None, str | None]:
    try:
        raw = json.loads(_cache_path(key).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(raw, dict):
        return None, None
    tag = raw.get("etag")
    return raw.get("body"), tag if isinstance(tag, str) else None


def _write_cache(key: str, body: Any, etag: str | None) -> None:
    try:
        _cache_dir().mkdir(parents=True, exist_ok=True)
        path = _cache_path(key)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"body": body, "etag": etag}), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass  # An uncacheable response is still a usable one.


def _cached_get(key: str, url: str, app_version: str, refresh: bool) -> tuple[Any | None, bool]:
    """``(body, stale)``. ``stale`` means the network failed and this is the saved copy.

    Always revalidates rather than serving the cache outright: the catalogue gains workflows and the
    popup has no refresh button, so a cache-first read would pin the first list a user ever saw. The
    stored ETag makes the usual case a 304, and ``refresh`` drops it to force a full body.
    """
    cached, etag = _read_cache(key)

    fetched = _fetch(url, app_version, None if refresh else etag)
    if fetched is None:
        return cached, True
    if fetched.unchanged:
        return cached, False

    _write_cache(key, fetched.body, fetched.etag)
    return fetched.body, False


def list_workflows(sort: str = "views", refresh: bool = False, app_version: str = "") -> dict[str, Any]:
    """The catalogue for the Workflows popup: entries plus the categories the rail is built from."""
    query = urlencode({"sort": sort, "limit": 200})
    body, stale = _cached_get(
        f"list-{sort}", f"{catalogue_url()}/api/workflows?{query}", app_version, refresh
    )
    if not isinstance(body, dict):
        return {"entries": [], "categories": [], "stale": True}

    entries = body.get("entries") or []
    for entry in entries:
        # Filled in only when the site did not send one: a deploy that predates the card link, or a
        # cached payload written before it. The site's own value always wins.
        if isinstance(entry, dict) and not entry.get("pageUrl") and entry.get("slug"):
            entry["pageUrl"] = f"{catalogue_url()}/workflows/{entry['slug']}"

    return {
        "entries": entries,
        "categories": body.get("categories") or [],
        "stale": stale,
    }


def workflow_detail(slug: str, app_version: str = "") -> dict[str, Any] | None:
    """One workflow with its graph. Uncached upstream, so every fetch counts as a view.

    The disk copy is a fallback only: it is written after a successful fetch and read only when the
    network fails, so a cached detail never suppresses a view that did reach the site.
    """
    fetched = _fetch(f"{catalogue_url()}/api/workflows/{slug}", app_version, None)
    if fetched is not None and isinstance(fetched.body, dict):
        _write_cache(f"detail-{slug}", fetched.body, None)
        return {**fetched.body, "stale": False}

    cached, _ = _read_cache(f"detail-{slug}")
    if isinstance(cached, dict):
        return {**cached, "stale": True}
    return None


def record_event(slug: str, event: str, app_version: str = "") -> None:
    """Beacon an event the site cannot observe on its own, chiefly a completed import."""
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    payload = json.dumps({"event": event}).encode("utf-8")
    request = Request(
        f"{catalogue_url()}/api/workflows/{slug}/events",
        data=payload,
        headers={**_headers(app_version), "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=_TIMEOUT):  # noqa: S310 - URL comes from config
            pass
    except (HTTPError, URLError, OSError):
        pass  # A counter that did not move must never surface as an error in the app.
