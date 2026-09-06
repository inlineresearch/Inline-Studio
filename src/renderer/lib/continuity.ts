/** Display helpers for a take's continuity score (0-100 identity match against a character). */
import type { TakeContinuity } from '@shared/types'

/** Coarse bands on purpose: the number is a comparison aid between takes, not a verdict. */
export function scoreTone(score: number): string {
  if (score >= 75) return 'text-emerald-300'
  if (score >= 50) return 'text-amber-300'
  return 'text-red-300'
}

export function scoreTitle(
  score: number,
  faceOnly = false,
  subjectOnly = false,
  wardrobe?: number,
): string {
  const base = `Continuity ${Math.round(score)} of 100 against the applied character`
  // Appended rather than blended: it answers a different question, and the two disagree often
  // enough that averaging them would hide both.
  const dressed =
    wardrobe === undefined
      ? ''
      : ` Wardrobe ${Math.round(wardrobe)} of 100 against the character's clothing references.`
  // No face at all is a different claim from a face measured without its subject term, and the
  // second one is the one that must not read as an identity match.
  if (subjectOnly) {
    return (
      `${base}. No face was found, so this is the subject term alone: it compares framing and
setting, not identity. Treat it as unmeasured.`.replace(/\s+/g, ' ') + dressed
    )
  }
  if (!faceOnly) return base + dressed
  // Say which term was dropped and how to get it back, not hide noise behind a total.
  return (
    `${base}. Face only: the references are all closely cropped, so they cannot judge this
framing. Add a full-body reference to score the whole subject.`.replace(/\s+/g, ' ') + dressed
  )
}

/** A short marker for the score pill, so a face-only number is not read as a whole-subject one. */
export function scoreSuffix(faceOnly = false, subjectOnly = false): string {
  if (subjectOnly) return ' no face'
  return faceOnly ? ' face' : ''
}

/** Read a take's continuity off wherever it is stored: a Core take ref carries the fields directly,
 *  a fal take carries them in its params blob, because the two paths persist metadata differently. */
export function takeContinuity(take: { params?: Record<string, unknown> }): TakeContinuity | null {
  const raw = take.params?.continuity
  return raw && typeof raw === 'object' ? (raw as TakeContinuity) : null
}

/** The minimum, not the mean: a clip averaging high while dipping low has a visible break. */
export function clipScoreLabel(c: TakeContinuity): string | null {
  if (c.continuityScore === undefined) return null
  const head = String(Math.round(c.continuityScore))
  if (c.continuityMin === undefined) return head + scoreSuffix(c.continuityFaceOnly)
  const at = c.continuityMinAt === undefined ? '' : ` @${c.continuityMinAt.toFixed(1)}s`
  return `${head} · min ${Math.round(c.continuityMin)}${at}`
}

/** The clip pill's tone, taken from the worst frame rather than the headline - the dip is the risk. */
export function clipScoreTone(c: TakeContinuity): string {
  return scoreTone(c.continuityMin ?? c.continuityScore ?? 0)
}

export function clipScoreTitle(c: TakeContinuity): string {
  const parts: string[] = []
  if (c.continuityMean !== undefined) parts.push(`Held at ${Math.round(c.continuityMean)} of 100`)
  else if (c.continuityScore !== undefined)
    parts.push(`Continuity ${Math.round(c.continuityScore)} of 100`)
  if (c.continuityMin !== undefined) {
    const at = c.continuityMinAt === undefined ? '' : ` around ${c.continuityMinAt.toFixed(1)}s`
    parts.push(`dropped to ${Math.round(c.continuityMin)}${at}`)
  }
  const measured = c.continuityFrames ?? 0
  const blind = c.continuityNoFace ?? 0
  if (blind > 0) {
    // Named rather than folded into the score: a turned head is not a wrong face, and counting it
    // as one would make every clip with natural movement look like a failure.
    parts.push(`${blind} of ${measured + blind} sampled frames had no face and were not scored`)
  } else if (measured > 0) {
    parts.push(`across ${measured} sampled frames`)
  }
  // Thresholds are uncalibrated on hosted output, so this says what was measured, never a verdict.
  return `${parts.join(', ')}.`
}
