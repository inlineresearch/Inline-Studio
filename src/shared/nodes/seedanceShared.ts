/**
 * Shared params + request/price helpers for ByteDance Seedance 2.0 (text / image / reference → video).
 * The three Seedance endpoints take the same generation controls; only their inputs differ.
 */
import { approxPrice, booleanParam, selectParam, seedParam, putSeed } from './builders'
import type { ParamField, PriceEstimate } from './types'

const RESOLUTIONS = ['480p', '720p', '1080p'] as const
// Seedance's `duration` is a string: "auto" or a whole number of seconds (4–15).
const DURATIONS = ['auto', '4', '5', '6', '8', '10', '12'] as const
const ASPECT_RATIOS = ['auto', '21:9', '16:9', '4:3', '1:1', '3:4', '9:16'] as const

/** ~USD per second of generated video by resolution (fal standard tier, approximate). */
const RATE_PER_SECOND: Record<string, number> = { '480p': 0.15, '720p': 0.3, '1080p': 0.68 }

export const SEEDANCE_PARAMS: ParamField[] = [
  selectParam('resolution', 'Resolution', RESOLUTIONS, '720p'),
  selectParam('duration', 'Duration (s)', DURATIONS, 'auto'),
  selectParam('aspect_ratio', 'Aspect ratio', ASPECT_RATIOS, 'auto'),
  booleanParam('generate_audio', 'Generate audio', true),
  seedParam(),
]

/** Build a Seedance request body: shared controls + `extra` input fields (image_url, image_urls, …). */
export function buildSeedanceBody(
  params: Record<string, unknown>,
  extra: Record<string, unknown>,
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    prompt: String(params.prompt ?? ''),
    resolution: params.resolution ?? '720p',
    duration: params.duration ?? 'auto',
    aspect_ratio: params.aspect_ratio ?? 'auto',
    generate_audio: Boolean(params.generate_audio ?? true),
    ...extra,
  }
  putSeed(body, params)
  return body
}

export function estimateSeedancePrice(params: Record<string, unknown>): PriceEstimate {
  const rate = RATE_PER_SECOND[String(params.resolution ?? '720p')] ?? RATE_PER_SECOND['720p']
  const d = String(params.duration ?? 'auto')
  const seconds = d === 'auto' ? 5 : Number(d) || 5
  return approxPrice(rate * seconds)
}
