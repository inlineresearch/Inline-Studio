"""The director-node timeline: resolve a wired timeline from the canvas, then render it with ffmpeg.

Ported from the Studio ``electron/main/{timeline,export,media}`` modules. ``compose`` builds the
ffmpeg arg vector (pure), ``ffmpeg`` runs it, ``resolve`` derives the timeline from the moodboard
connectors, and ``render`` orchestrates preview/export.
"""
