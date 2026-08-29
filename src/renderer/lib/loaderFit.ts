// Sizes a "Load Assets" node to its media once, when its first asset lands.

/** The node's footer bar; the fit runs before mount, so it cannot be measured off the DOM. */
export const LOADER_CHROME_H = 26
/** Both orientations get the same visual weight by matching on their long edge. */
export const LOADER_LONG_EDGE = 340
export const LOADER_MIN_W = 200
export const LOADER_MAX_W = 460
export const LOADER_MIN_BODY = 150
export const LOADER_MAX_BODY = 460

const clamp = (n: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, n))

/** Node size for a media aspect; extreme ratios clamp and keep a blurred sliver. */
export function fitLoaderSize(aspect: number): { width: number; height: number } {
  // Callers measure first, so this only keeps NaN out of the stored width/height.
  const safe = Number.isFinite(aspect) && aspect > 0 ? aspect : 1
  const long = safe >= 1 ? LOADER_LONG_EDGE : LOADER_LONG_EDGE * safe
  const width = clamp(Math.round(long), LOADER_MIN_W, LOADER_MAX_W)
  const body = clamp(Math.round(width / safe), LOADER_MIN_BODY, LOADER_MAX_BODY)
  return { width, height: body + LOADER_CHROME_H }
}

/** Whether a box of this aspect leaves visible bars around media of that aspect. */
export function needsBlurFill(mediaAspect: number, boxAspect: number): boolean {
  if (!Number.isFinite(mediaAspect) || !Number.isFinite(boxAspect)) return false
  if (mediaAspect <= 0 || boxAspect <= 0) return false
  return Math.abs(mediaAspect - boxAspect) / boxAspect > 0.02
}
