/** Display helpers for a take's continuity score (0-100 identity match against a character). */

/** Coarse bands on purpose: the number is a comparison aid between takes, not a verdict. */
export function scoreTone(score: number): string {
  if (score >= 75) return 'text-emerald-300'
  if (score >= 50) return 'text-amber-300'
  return 'text-red-300'
}

export function scoreTitle(score: number, faceOnly = false): string {
  const base = `Continuity ${Math.round(score)} of 100 against the applied character`
  if (!faceOnly) return base
  // Say which term was dropped and how to get it back, not hide noise behind a total.
  return `${base}. Face only: the references are all closely cropped, so they cannot judge this
framing. Add a full-body reference to score the whole subject.`.replace(/\s+/g, ' ')
}

/** A short marker for the score pill, so a face-only number is not read as a whole-subject one. */
export function scoreSuffix(faceOnly = false): string {
  return faceOnly ? ' face' : ''
}
