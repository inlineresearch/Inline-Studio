/**
 * MiniMax H3's clip frame grid, mirrored from Core so the Trainer can show what a clip length
 * actually resolves to before a run starts.
 *
 * H3's video VAE only encodes `17n + 5` frames at 24fps, and Core snaps **down** onto that grid
 * (`trim_reference_num_frames`). Asking for 5s therefore trains on 4.458s, and the shortfall comes
 * off the end. That was silent until the number appeared next to the field, and a user lost the
 * action in the last half second of their clips to it.
 */

/** H3's fixed frame rate. Never a parameter: it is what the grid is defined against. */
export const H3_FPS = 24

/** The shortest clip the video VAE encodes: the first `17n + 5` above a single frame. */
export const H3_MIN_CLIP_FRAMES = 22

/** The largest `17n + 5` that fits `frames`, or `H3_MIN_CLIP_FRAMES` when nothing does. */
export function snapClipFrames(frames: number): number {
  if (!Number.isFinite(frames) || frames < H3_MIN_CLIP_FRAMES) return H3_MIN_CLIP_FRAMES
  return Math.floor((frames - 5) / 17) * 17 + 5
}

/** What `seconds` of clip actually trains on, as `{frames, seconds}` on the grid. */
export function resolveClipLength(seconds: number): { frames: number; seconds: number } {
  const frames = snapClipFrames(Math.round(seconds * H3_FPS))
  return { frames, seconds: frames / H3_FPS }
}
