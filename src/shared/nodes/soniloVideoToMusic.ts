/**
 * fal.ai `sonilo/v1.1/video-to-music` — generates an original soundtrack matched to an input video.
 * See https://fal.ai/models/sonilo/v1.1/video-to-music.
 *
 * The first audio-output model in the registry: it proves the registry generalizes to audio
 * (video input → audio output), and the resulting audio frame wires straight into a Video
 * Director's L2 (music) inputs. The prompt is optional — when none is connected, Sonilo reads
 * the video's pacing and mood and writes its own. Outputs are licensed and safe for commercial
 * use (terms apply — see the fal.ai model page).
 */
import { type NodeDef, type ResolvedInputs } from './types'
import { constantEndpoint, approxPrice, numberParam } from './builders'

const ENDPOINT = 'sonilo/v1.1/video-to-music'

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

export const SONILO_V2M: NodeDef = {
  id: 'sonilo/v1.1/video-to-music',
  title: 'Sonilo · Video → Music',
  category: 'Audio',
  provider: 'fal',
  outputKind: 'audio',
  inputs: [{ id: 'video', label: 'Video', kind: 'video', required: true }],
  // The prompt only steers the musical style, so it's optional — Sonilo generates one from the
  // video when nothing is wired (unlike every other model, where the prompt IS the instruction).
  promptOptional: true,
  params: [
    numberParam('num_samples', 'Samples', 1, { min: 1, step: 1 }),
    numberParam('start_offset', 'Start offset (s)', 0, { min: 0, step: 1 }),
    numberParam('duration', 'Duration (s, 0 = full video)', 0, { min: 0, step: 1 }),
  ],
  outputs: [{ id: 'audio', label: 'Music', kind: 'audio' }],

  resolveEndpoint: constantEndpoint(ENDPOINT),

  buildRequest(params: Record<string, unknown>, resolved: ResolvedInputs): Record<string, unknown> {
    const body: Record<string, unknown> = {
      video_url: resolved.videos[0] ?? '',
      num_samples: toSamples(params.num_samples),
    }
    const prompt = String(params.prompt ?? '').trim()
    if (prompt) body.prompt = prompt
    const startOffset = toSeconds(params.start_offset)
    if (startOffset !== null) body.start_offset = startOffset
    const duration = toSeconds(params.duration)
    if (duration !== null) body.duration = duration
    return body
  },

  // fal bills $0.009 per second of output audio, per sample. The output runs the length of the
  // selected segment; without an explicit duration the video's length isn't known here, so the
  // estimate only shows once a segment duration is set.
  // Source: fal.ai/models/sonilo/v1.1/video-to-music.
  estimatePrice(params) {
    const duration = toSeconds(params.duration)
    if (duration === null) return null
    return approxPrice(duration * toSamples(params.num_samples) * 0.009)
  },
}
