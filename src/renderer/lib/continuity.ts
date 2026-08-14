/** Display helpers for a take's continuity score (0-100 identity match against a character). */

/** Coarse bands on purpose: the number is a comparison aid between takes, not a verdict. */
export function scoreTone(score: number): string {
  if (score >= 75) return 'text-emerald-300'
  if (score >= 50) return 'text-amber-300'
  return 'text-red-300'
}

export function scoreTitle(score: number): string {
  return `Continuity ${Math.round(score)} of 100 against the applied character`
}
