/**
 * What a character build resolves to before it runs: which step counts are offered, and roughly how
 * long each takes. Every number here was measured, never estimated, because a duration in the UI is
 * a promise the run has to keep.
 */

/** Only values with a measured score behind them. 1200 peaked; 2000 regressed on faces. */
export const BUILD_STEP_CHOICES = [600, 1200, 2000] as const

export type BuildSteps = (typeof BUILD_STEP_CHOICES)[number]

/** Seconds per training step, measured on an L4 at rank 16, resolution 512, full LoRA scope. */
export const SECONDS_PER_STEP: Record<string, number> = {
  // 600 steps in 9.7 min, 1200 in 19.4, 2000 in 32.3, all on klein base 4B.
  flux2: 0.97,
  // 600 steps in 61.5 min on the turbo-plus-adapter path.
  krea2: 6.15,
}

/** The hardware the numbers above came from, so a hint can say what it is basing itself on. */
export const TIMING_BASIS = 'measured on an L4'

/** How each step count scored, so the UI can say why 2000 is not simply "better". */
export const STEP_NOTES: Record<number, string> = {
  600: 'fast',
  1200: 'best on faces',
  2000: 'worse on faces, better on body',
}

/** A rough wall time for a build, or null when the architecture has never been timed. */
export function estimateMinutes(arch: string, steps: number): number | null {
  const perStep = SECONDS_PER_STEP[arch]
  if (!perStep) return null
  return Math.round((perStep * steps) / 60)
}

/** "~19 min" / "~2 h 3 min", or null when there is no measurement to base it on. */
export function formatEstimate(arch: string, steps: number): string | null {
  const minutes = estimateMinutes(arch, steps)
  if (minutes === null) return null
  if (minutes < 60) return `~${minutes} min`
  const hours = Math.floor(minutes / 60)
  const rest = minutes % 60
  return rest === 0 ? `~${hours} h` : `~${hours} h ${rest} min`
}
