/**
 * Shared params + request/price helpers for MiniMax H3 (Hailuo 03): text / image / reference → video.
 * The three endpoints take the same generation controls; only their inputs and framing differ.
 */
import { approxPrice, numberParam, selectParam } from './builders'
import type { ParamField, PriceEstimate } from './types'

// H3 is 2K-only - fal's schema pins `resolution` to the literal "2K". Kept as a one-option select so
// the node reads the same as fal's own form and the price rate below has a visible source.
const RESOLUTIONS = ['2K'] as const

export const H3_ASPECT_RATIOS = ['21:9', '16:9', '4:3', '1:1', '3:4', '9:16'] as const

/** USD per second of 2K output. Reference surcharges are extra; see `estimateH3Price`. */
const RATE_PER_SECOND = 0.26

const MIN_DURATION = 5
const MAX_DURATION = 15

/** Resolution + duration, shared by all three H3 endpoints. Aspect ratio is per-endpoint. */
export const H3_BASE_PARAMS: ParamField[] = [
  selectParam('resolution', 'Resolution', RESOLUTIONS, '2K'),
  numberParam('duration', `Duration (${MIN_DURATION} to ${MAX_DURATION}s)`, MIN_DURATION, {
    min: MIN_DURATION,
    max: MAX_DURATION,
    step: 1,
  }),
]

export function h3Duration(params: Record<string, unknown>): number {
  const raw = Math.round(Number(params.duration ?? MIN_DURATION))
  if (!Number.isFinite(raw)) return MIN_DURATION
  return Math.min(MAX_DURATION, Math.max(MIN_DURATION, raw))
}

/** Build an H3 request body: shared controls + `extra` input/framing fields. */
export function buildH3Body(
  params: Record<string, unknown>,
  extra: Record<string, unknown>,
): Record<string, unknown> {
  return {
    prompt: String(params.prompt ?? ''),
    duration: h3Duration(params),
    resolution: '2K',
    ...extra,
  }
}

/**
 * Base cost only. Reference-to-video also bills for images past the first five and for reference
 * video, but `estimatePrice` sees params and not the wired inputs, so those cannot be counted here.
 */
export function estimateH3Price(params: Record<string, unknown>): PriceEstimate {
  return approxPrice(RATE_PER_SECOND * h3Duration(params))
}
