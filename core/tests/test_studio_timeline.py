"""Timeline: the pure ffmpeg-arg builder, connector resolution, and the hero-take folder export."""

from __future__ import annotations

from inline_core.studio import frames as fr
from inline_core.studio import moodboard as mb
from inline_core.studio.store import StudioStore
from inline_core.studio.timeline.compose import (
    ComposeSettings,
    ResolvedClip,
    build_compose_args,
    timeline_duration,
)
from inline_core.studio.timeline.render import export_frames
from inline_core.studio.timeline.resolve import resolve_timeline


def _settings(out="out.mp4") -> ComposeSettings:
    return ComposeSettings(width=1920, height=1080, fps=30, preset="veryfast", crf=20, out_path=out)


def test_timeline_duration() -> None:
    clips = [
        ResolvedClip("image", "/a.png", 0, 0.0, 0.0, 4.0),
        ResolvedClip("video", "/b.mp4", 0, 4.0, 0.0, 6.0),
    ]
    assert timeline_duration(clips) == 10.0
    assert timeline_duration([]) == 0.04


def test_build_compose_args_structure() -> None:
    clips = [
        ResolvedClip("image", "/a.png", 0, 0.0, 0.0, 4.0),
        ResolvedClip("video", "/b.mp4", 0, 4.0, 0.0, 6.0, has_audio=True, volume=0.5),
        ResolvedClip("audio", "/c.mp3", 1, 0.0, 0.0, 8.0, volume=0.8),
    ]
    args = build_compose_args(clips, _settings())
    joined = " ".join(args)
    # Base black + silent beds, then one input per clip.
    assert "color=c=black:s=1920x1080:r=30" in joined and "anullsrc" in joined
    assert "-loop 1" in joined  # the image clip loops
    assert args.count("-i") == 5  # 2 beds + 3 clips
    # Video overlay + audio mix filtergraph present, and libx264/aac output.
    fc = args[args.index("-filter_complex") + 1]
    assert "overlay=enable" in fc and "amix=inputs=" in fc and "volume=0.50" in fc
    assert args[-1] == "out.mp4" and "libx264" in args and "aac" in args


def test_build_compose_args_empty_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        build_compose_args([], _settings())


def test_resolve_timeline_from_connectors(tmp_path) -> None:
    store = StudioStore(tmp_path / "app", tmp_path / "ws")
    store.create_project("Film")
    conn, folder = store.conn(), store.folder()
    # A frame with an image hero take on disk, placed as a canvas node.
    frame_item = mb.add_empty_frame(conn, 0, 0)
    (folder / "takes").mkdir(exist_ok=True)
    (folder / "takes" / "t.png").write_bytes(b"\x89PNG")
    fr.add_take(conn, frame_item["frameId"], "takes/t.png", "image", {})
    # A director node with the frame wired into its video slot 0.
    director = mb.add_director(conn, 500, 0)
    mb.create_connector(conn, frame_item["id"], director["id"], "out", "vin-0")

    display, clips = resolve_timeline(conn, folder, director["id"])
    assert len(display["video"]) == 1 and len(clips) == 1
    assert display["video"][0]["kind"] == "image"
    assert clips[0].track == 0 and clips[0].kind == "image"
    assert display["l1Volume"] == 1 and display["l2"] == []


def test_real_ffmpeg_render_of_a_still(tmp_path) -> None:
    """End-to-end: a 1-image timeline actually renders to a valid MP4 (skipped without ffmpeg)."""
    import asyncio

    import pytest

    from inline_core.studio.timeline.ffmpeg import compose_render, ffmpeg_available, ffmpeg_exe

    if not ffmpeg_available():
        pytest.skip("ffmpeg not installed")

    async def run() -> tuple[bool, int]:
        img = tmp_path / "red.png"
        proc = await asyncio.create_subprocess_exec(
            ffmpeg_exe(), "-y", "-f", "lavfi", "-i", "color=c=red:s=64x64", "-frames:v", "1",
            str(img), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        out = tmp_path / "out.mp4"
        clips = [ResolvedClip("image", str(img), 0, 0.0, 0.0, 1.0)]
        settings = ComposeSettings(64, 64, 24, "ultrafast", 30, str(out))
        ok = await compose_render(build_compose_args(clips, settings), 1.0, lambda f: None)
        return ok, out.stat().st_size if out.is_file() else 0

    ok, size = asyncio.run(run())
    assert ok is True and size > 0


def test_export_frames_copies_hero_takes(tmp_path) -> None:
    store = StudioStore(tmp_path / "app", tmp_path / "ws")
    store.create_project("Film")
    conn, folder = store.conn(), store.folder()
    (folder / "takes").mkdir(exist_ok=True)
    f1 = fr.create_empty_frame(conn)
    (folder / "takes" / "a.png").write_bytes(b"img1")
    fr.add_take(conn, f1["id"], "takes/a.png", "image", {})
    fr.create_empty_frame(conn)  # a second frame with no output -> skipped

    result = export_frames(conn, folder)
    assert result["exported"] == 1 and len(result["skipped"]) == 1
    from pathlib import Path

    files = list(Path(result["dir"]).iterdir())
    assert len(files) == 1 and files[0].read_bytes() == b"img1"
