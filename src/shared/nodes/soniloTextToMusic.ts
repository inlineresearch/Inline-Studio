/**
 * fal.ai `sonilo/v1.1/text-to-music` - generates an original track from a text description alone
 * (no video input). See https://fal.ai/models/sonilo/v1.1/text-to-music.
 *
 * The video-to-music sibling reads a cut and writes to it; this one writes from words. The prompt
 * carries the whole instruction here, so - unlike video-to-music - it stays required: describe the
 * track (mood, instrumentation, tempo) and Sonilo returns audio takes. Outputs are licensed and
 * safe for commercial use (terms apply - see the fal.ai model page). It rides the same fal
 * transport and fal key as every other fal node.
 */
import { type NodeDef } from './types'
import { constantEndpoint, approxPrice, numberParam } from './builders'

const ENDPOINT = 'sonilo/v1.1/text-to-music'

/** Coerce a param to a sample count of at least 1 (fal default). */
function toSamples(value: unknown): number {
  const n = Math.round(Number(value ?? 1))
  return Number.isFinite(n) && n >= 1 ? n : 1
}

/** Coerce the duration to a positive second count, clamped to the endpoint's 600 s ceiling. */
function toDuration(value: unknown): number {
  const n = Math.round(Number(value ?? 90))
  if (!Number.isFinite(n) || n < 1) return 90
  return Math.min(n, 600)
}

export const SONILO_T2M: NodeDef = {
  id: 'sonilo/v1.1/text-to-music',
  title: 'Sonilo · Text → Music',
  category: 'Audio',
  provider: 'fal',
  outputKind: 'audio',
  // No media input - the prompt is the whole instruction, so it stays required (the default).
  inputs: [],
  params: [
    numberParam('duration', 'Duration (s)', 90, { min: 1, max: 600, step: 1 }),
    numberParam('num_samples', 'Samples', 1, { min: 1, step: 1 }),
  ],
  outputs: [{ id: 'audio', label: 'Music', kind: 'audio' }],

  resolveEndpoint: constantEndpoint(ENDPOINT),

  buildRequest(params: Record<string, unknown>): Record<string, unknown> {
    return {
      prompt: String(params.prompt ?? '').trim(),
      duration: toDuration(params.duration),
      num_samples: toSamples(params.num_samples),
    }
  },

  // fal bills $0.009 per second of output audio, per sample. Duration is always known here (it's a
  // param), so the estimate always shows. Source: fal.ai/models/sonilo/v1.1/text-to-music.
  estimatePrice(params) {
    return approxPrice(toDuration(params.duration) * toSamples(params.num_samples) * 0.009)
  },
}
