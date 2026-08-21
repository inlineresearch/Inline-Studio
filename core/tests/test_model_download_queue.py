"""Downloads run one at a time and can be stopped, so "download all" is answerable."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from inline_core.studio.models import DownloadCancelled, ModelDownloads


class _Events:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict]] = []

    def broadcast(self, channel: str, payload: dict) -> None:
        self.sent.append((channel, payload))


def _downloads(monkeypatch: pytest.MonkeyPatch) -> tuple[ModelDownloads, _Events, list[str]]:
    events = _Events()
    downloads = ModelDownloads(events)
    started: list[str] = []

    # Stand in for the thread that moves bytes: record the id and leave the slot held.
    def fake_run(model_id: str, _loop: Any) -> None:
        started.append(model_id)

    monkeypatch.setattr(downloads, "_run_registry", fake_run)
    monkeypatch.setattr(
        downloads, "_emit", lambda _loop, ch, payload: events.broadcast(ch, payload)
    )
    return downloads, events, started


def test_one_downloads_and_the_rest_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Several at once share the same link, so each lands later than if it had waited."""
    downloads, events, started = _downloads(monkeypatch)

    async def main() -> None:
        downloads._active = "already-running"
        downloads.download_registry("a")
        downloads.download_registry("b")
        downloads.download_registry("c")

    asyncio.run(main())
    assert downloads.download_queue()["queued"] == ["a", "b", "c"]
    queued = [p for _c, p in events.sent if str(p.get("status", "")).startswith("Queued")]
    assert [p["componentId"] for p in queued] == ["a", "b", "c"]
    assert queued[2]["status"] == "Queued (3)", "the wait says how many are ahead"


def test_asking_twice_does_not_queue_it_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    downloads, _events, _started = _downloads(monkeypatch)

    async def main() -> None:
        downloads._active = "already-running"
        downloads.download_registry("a")
        downloads.download_registry("a")

    asyncio.run(main())
    assert downloads.download_queue()["queued"] == ["a"]


def test_cancelling_a_waiting_download_drops_it(monkeypatch: pytest.MonkeyPatch) -> None:
    downloads, events, _started = _downloads(monkeypatch)

    async def main() -> None:
        downloads._active = "already-running"
        downloads.download_registry("a")
        downloads.download_registry("b")

    asyncio.run(main())
    downloads.cancel_registry("b")

    assert downloads.download_queue()["queued"] == ["a"]
    assert ("events:modelDownloadError", {
        "nodeType": "registry", "componentId": "b", "error": "Cancelled."
    }) in events.sent


def test_cancelling_the_running_one_stops_it_at_the_next_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A download has no cancel of its own; its progress callback is the only per-chunk seam."""
    downloads, _events, _started = _downloads(monkeypatch)
    loop = asyncio.new_event_loop()
    try:
        downloads._active = "a"
        # Still running: progress is forwarded.
        downloads._registry_progress(loop, "a", 0.5, "Downloading…")

        downloads.cancel_registry("a")
        with pytest.raises(DownloadCancelled):
            downloads._registry_progress(loop, "a", 0.6, "Downloading…")
    finally:
        loop.close()


def test_the_queue_advances_when_a_download_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tail of a download runs on a worker thread, which has no running loop for create_task to
    attach to: the first model landed, then every queued one sat at "Queued" for ever."""
    events = _Events()
    downloads = ModelDownloads(events)
    started: list[str] = []

    # The real _run_registry, so its finally clause is what has to hand the slot on.
    monkeypatch.setattr(
        downloads, "_run_registry_inner", lambda model_id, _loop: started.append(model_id)
    )
    monkeypatch.setattr(
        downloads, "_emit", lambda _loop, ch, payload: events.broadcast(ch, payload)
    )

    async def main() -> None:
        downloads.download_registry("a")
        downloads.download_registry("b")
        for _ in range(300):
            if len(started) == 2 and downloads.download_queue()["active"] is None:
                return
            await asyncio.sleep(0.01)

    asyncio.run(main())

    assert started == ["a", "b"], "the second ran only because the first advanced the queue"
    assert downloads.download_queue() == {"active": None, "queued": []}


def test_named_files_and_a_named_subfolder_are_separate_cases(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mixing them fetched the files to the staging root and then looked for them under a subfolder
    nothing had written: "No such file or directory: .../dinov2-base.part/dinov2-base"."""
    from inline_core.models.requirements import ModelComponent
    from inline_core.studio.models import _TargetOnly

    events = _Events()
    downloads = ModelDownloads(events)
    seen: dict[str, Any] = {}

    def fake_snapshot(repo: str, **kw: Any) -> str:
        seen.update(kw)
        root = kw["local_dir"]
        import os

        os.makedirs(root, exist_ok=True)
        for name in kw["allow_patterns"]:
            open(f"{root}/{name}", "w").close()
        return root

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    comp = ModelComponent(
        id="dinov2", label="Subject embeddings", category="annotators", present=False,
        filename="dinov2-base", repo="facebook/dinov2-base", repo_file="",
        repo_files=("config.json", "model.safetensors"),
    )
    downloads._download_component(_TargetOnly(tmp_path), comp, lambda _f, _s: None)

    landed = tmp_path / "dinov2-base"
    assert landed.is_dir(), "the folder is what the node loads from"
    assert (landed / "model.safetensors").is_file()
    assert seen["allow_patterns"] == ["config.json", "model.safetensors"]
