/**
 * fal.ai `sonilo/v1.1/video-to-video` — returns the input video with a generated, licensed track
 * mixed in. See https://fal.ai/models/sonilo/v1.1/video-to-video.
 *
 * The video-to-music sibling hands back an audio take you wire into the Video Director; this one
 * hands back the finished video, so it slots in when you want the mix done in one step. Set
 * `keep_speech_vocal` to preserve the original speech alongside the new track. Like video-to-music
 * the prompt is optional — Sonilo reads the video's pacing and mood when nothing is wired, and a
 * prompt only steers the musical style. Outputs are licensed and safe for commercial use (terms
 * apply — see the fal.ai model page). It rides the same fal transport and fal key as every other
 * fal node.
 */
import { type NodeDef, type ResolvedInputs } from './types'
import { constantEndpoint, numberParam, booleanParam, approxPrice } from './builders'

const ENDPOINT = 'sonilo/v1.1/video-to-video'

/** Coerce a param to a sample count of at least 1 (fal default). */
function toSamples(value: unknown): number {
  const n = Math.round(Number(value ?? 1))
  return Number.isFinite(n) && n >= 1 ? n : 1
}

/** Coerce an optional seconds param; null when unset/zero (the field is omitted from the request). */
function toSeconds(value: unknown): number | null {
  const n = Number(value ?? 0)
  return Number.isFinite(n) && n > 0 ? n : null
}

export const SONILO_V2V: NodeDef = {
  id: 'sonilo/v1.1/video-to-video',
  title: 'Sonilo · Video → Video',
  category: 'Video',
  provider: 'fal',
  outputKind: 'video',
  inputs: [{ id: 'video', label: 'Video', kind: 'video', required: true }],
  // Like video-to-music: the prompt only steers the musical style, so it's optional — Sonilo
  // reads the video and writes its own when nothing is wired.
  promptOptional: true,
  params: [
    // "Preserve original speech": when on, Sonilo separates the source audio and blends the
    // original vocals back over the generated track (fal field: keep_speech_vocal).
    booleanParam('keep_speech_vocal', 'Keep original speech', false, { advanced: false }),
    numberParam('num_samples', 'Samples', 1, { min: 1, step: 1 }),
    numberParam('start_offset', 'Start offset (s)', 0, { min: 0, step: 1 }),
    numberParam('duration', 'Duration (s, 0 = full video)', 0, { min: 0, step: 1 }),
  ],
  outputs: [{ id: 'video', label: 'Video', kind: 'video' }],

  resolveEndpoint: constantEndpoint(ENDPOINT),

  buildRequest(params: Record<string, unknown>, resolved: ResolvedInputs): Record<string, unknown> {
    const body: Record<string, unknown> = {
      video_url: resolved.videos[0] ?? '',
      num_samples: toSamples(params.num_samples),
      keep_speech_vocal: Boolean(params.keep_speech_vocal),
    }
    const prompt = String(params.prompt ?? '').trim()
    if (prompt) body.prompt = prompt
    const startOffset = toSeconds(params.start_offset)
    if (startOffset !== null) body.start_offset = startOffset
    const duration = toSeconds(params.duration)
    if (duration !== null) body.duration = duration
    return body
  },

  // fal bills $0.009 per second of the mixed output, per sample. Without an explicit duration the
  // video's length isn't known here, so the estimate only shows once a segment duration is set.
  // Source: fal.ai/models/sonilo/v1.1/video-to-video.
  estimatePrice(params) {
    const duration = toSeconds(params.duration)
    if (duration === null) return null
    return approxPrice(duration * toSamples(params.num_samples) * 0.009)
  },
}
