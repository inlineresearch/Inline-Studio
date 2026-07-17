"""Build the ffmpeg args to render a director timeline (EDL) into one muxed MP4 — a faithful
port of the Studio ``electron/main/export/compose.ts``. Pure + deterministic, so unit-tested.

Model: clips reference absolute files at ``start_time``, trimmed to in/out. Track 0 = video
(images loop for a synthetic duration), track 1 = audio. Heterogeneous sources are normalised to a
common W×H/fps and 44.1k stereo, then overlaid / mixed onto a base black-video + silent-audio bed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResolvedClip:
    kind: str  # 'video' | 'image' | 'audio'
    abs_path: str
    track: int  # 0 = video, 1 = audio
    start_time: float
    in_point: float
    out_point: float
    has_audio: bool = False
    muted: bool = False
    volume: float = 1.0


@dataclass
class ComposeSettings:
    width: int
    height: int
    fps: int
    preset: str
    crf: int
    out_path: str


def _clip_duration(c: ResolvedClip) -> float:
    return max(0.04, c.out_point - c.in_point)


def _clip_end(c: ResolvedClip) -> float:
    return c.start_time + _clip_duration(c)


def timeline_duration(clips: list[ResolvedClip]) -> float:
    """Total timeline length = the furthest clip end (min 0.04s)."""
    return max((_clip_end(c) for c in clips), default=0.04)


def _contributes_audio(c: ResolvedClip) -> bool:
    if c.volume <= 0:
        return False
    if c.track == 1:
        return True
    return c.kind == "video" and not c.muted and c.has_audio


def build_compose_args(clips: list[ResolvedClip], s: ComposeSettings) -> list[str]:
    """The full ffmpeg arg vector (excluding the binary). Raises if there are no clips."""
    if not clips:
        raise ValueError("Cannot compose an empty timeline.")
    total_str = f"{timeline_duration(clips):.3f}"
    args: list[str] = ["-y"]

    # Base beds: a black video and silent audio spanning the whole timeline.
    args += ["-f", "lavfi", "-t", total_str, "-i",
             f"color=c=black:s={s.width}x{s.height}:r={s.fps}"]
    args += ["-f", "lavfi", "-t", total_str, "-i",
             "anullsrc=channel_layout=stereo:sample_rate=44100"]

    # One input per clip (index starts at 2). Images loop for their duration.
    for c in clips:
        dur = f"{_clip_duration(c):.3f}"
        if c.kind == "image":
            args += ["-loop", "1", "-t", dur, "-i", c.abs_path]
        else:
            args += ["-ss", f"{c.in_point:.3f}", "-t", dur, "-i", c.abs_path]

    filters: list[str] = []
    video_label = "0:v"
    for i, c in enumerate(clips):
        inp = i + 2
        if c.track == 0 and c.kind in ("video", "image"):
            v = f"v{i}"
            filters.append(
                f"[{inp}:v]scale={s.width}:{s.height}:force_original_aspect_ratio=decrease,"
                f"pad={s.width}:{s.height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={s.fps},"
                f"setpts=PTS-STARTPTS+{c.start_time:.3f}/TB[{v}]"
            )
            out = f"ov{i}"
            filters.append(
                f"[{video_label}][{v}]overlay="
                f"enable='between(t,{c.start_time:.3f},{_clip_end(c):.3f})'[{out}]"
            )
            video_label = out

    audio_labels: list[str] = ["1:a"]
    for i, c in enumerate(clips):
        inp = i + 2
        if _contributes_audio(c):
            ms = round(c.start_time * 1000)
            dur = _clip_duration(c)
            fade = min(0.02, dur / 4)
            fade_out = f"{max(0.0, dur - fade):.3f}"
            a = f"a{len(audio_labels) - 1}"
            filters.append(
                f"[{inp}:a]atrim=0:{dur:.3f},asetpts=PTS-STARTPTS,"
                f"aformat=sample_rates=44100:channel_layouts=stereo,"
                f"volume={c.volume:.2f},"
                f"afade=t=in:st=0:d={fade:.3f},afade=t=out:st={fade_out}:d={fade:.3f},"
                f"adelay={ms}|{ms}[{a}]"
            )
            audio_labels.append(a)

    audio_out = "1:a"
    if len(audio_labels) > 1:
        audio_out = "aout"
        joined = "".join(f"[{lbl}]" for lbl in audio_labels)
        filters.append(
            f"{joined}amix=inputs={len(audio_labels)}:normalize=0:dropout_transition=0[{audio_out}]"
        )

    if filters:
        args += ["-filter_complex", ";".join(filters)]
    args += ["-map", "0:v" if video_label == "0:v" else f"[{video_label}]"]
    args += ["-map", "1:a" if audio_out == "1:a" else f"[{audio_out}]"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", s.preset, "-crf", str(s.crf),
             "-c:a", "aac", "-movflags", "+faststart", "-r", str(s.fps), "-t", total_str,
             s.out_path]
    return args
