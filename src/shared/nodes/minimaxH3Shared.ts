/**
 * Shared params + request/price helpers for MiniMax H3 (Hailuo 03): text / image / reference → video.
 * The three endpoints take the same generation controls; only their inputs and framing differ.
 */
import { approxPrice, numberParam, selectParam } from './builders'
import type { ParamField, PriceEstimate, PriceInputs } from './types'

// All four fal documents for every H3 endpoint. This used to offer 2K alone, which cost a user the
// cheaper tiers: a draft at 480P is a fifth of the price of the 2K it was silently pinned to.
const RESOLUTIONS = ['480P', '768P', '2K', '4K'] as const

export const H3_ASPECT_RATIOS = ['21:9', '16:9', '4:3', '1:1', '3:4', '9:16'] as const

/** USD per second, by resolution, from fal's own pricing for H3. */
const RATE_PER_SECOND: Record<string, number> = {
  '480P': 0.05,
  '768P': 0.06,
  '2K': 0.13,
  '4K': 0.16,
}
const DEFAULT_RESOLUTION = '2K'

const MIN_DURATION = 5
const MAX_DURATION = 15

/** Resolution + duration, shared by all three H3 endpoints. Aspect ratio is per-endpoint. */
export const H3_BASE_PARAMS: ParamField[] = [
  selectParam('resolution', 'Resolution', RESOLUTIONS, DEFAULT_RESOLUTION),
  numberParam('duration', `Duration (${MIN_DURATION} to ${MAX_DURATION}s)`, MIN_DURATION, {
    min: MIN_DURATION,
    max: MAX_DURATION,
    step: 1,
  }),
]

/** The picked resolution, or the default when a node predates the param having options. */
export function h3Resolution(params: Record<string, unknown>): string {
  const raw = String(params.resolution ?? DEFAULT_RESOLUTION)
  return raw in RATE_PER_SECOND ? raw : DEFAULT_RESOLUTION
}

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
    resolution: h3Resolution(params),
    ...extra,
  }
}

/** The first five reference images are free; each one past that adds this much. */
const FREE_REFERENCE_IMAGES = 5
const RATE_PER_EXTRA_IMAGE = 0.08

export function estimateH3Price(
  params: Record<string, unknown>,
  wired?: PriceInputs,
): PriceEstimate {
  const seconds = RATE_PER_SECOND[h3Resolution(params)] * h3Duration(params)
  const extra = Math.max(0, (wired?.referenceImages ?? 0) - FREE_REFERENCE_IMAGES)
  return approxPrice(seconds + extra * RATE_PER_EXTRA_IMAGE)
}
