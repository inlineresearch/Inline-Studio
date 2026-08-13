/**
 * How comfortably each starter model will run on this machine.
 *
 * Advisory, never gating. The device policy auto-quantizes and offloads to fit, so "will not run"
 * is nearly always wrong; what a new user actually needs to know is which model will feel fast and
 * which will crawl. Every tier's copy therefore still says it runs.
 *
 * Tiering deliberately does NOT use the engine's own fit estimate. That sizes models from what is
 * on disk, so on a fresh install with nothing downloaded every model reports a zero footprint and
 * "fits, resident" - the exact opposite of useful. Static requirements are correct before the first
 * download, which is when this advice is read.
 */
import type { SystemStatsEvent } from '@shared/types'

export type StarterKey = 'ltx25' | 'minimaxh3' | 'zimage' | 'flux2' | 'krea2' | 'training'
export type Tier = 'best' | 'good' | 'ok' | 'heavy'

/** Every starter now runs on this machine, so VRAM has something to say about all of them. */
type LocalKey = StarterKey

export type VramReading =
  | { state: 'pending' }
  /** No GPU memory could be read. Ambiguous: an Apple Silicon or AMD box, or NVML missing. */
  | { state: 'unknown' }
  | { state: 'known'; totalGb: number; name: string }

/** Total VRAM in GiB, from the largest card present. */
export function readVram(stats: SystemStatsEvent | null): VramReading {
  if (stats == null) return { state: 'pending' }
  const gpus = stats.gpus ?? []
  if (gpus.length === 0) return { state: 'unknown' }
  // The largest card, not gpus[0]: a box with a small display GPU at index 0 would otherwise be
  // mis-tiered. The device policy may still choose device 0, which is fine for a recommendation.
  const largest = gpus.reduce((a, b) => (b.memoryTotal > a.memoryTotal ? b : a))
  return { state: 'known', totalGb: largest.memoryTotal / 1024 ** 3, name: largest.name }
}

interface TierRow {
  minGb: number
  tier: Tier
  note: string
}

/**
 * Descending thresholds per model, from measured footprints: Z-Image is 12.3 GB of weights plus an
 * 8 GB encoder that can be evicted after encoding; FLUX.2 klein 4B is 16.1 GB resident at bf16;
 * Krea 2 is around 26 GB and drops to NF4 below roughly 24 GB.
 *
 * A card advertising 16 GB reports slightly under 16 GiB, so the rows sit a little below the round
 * number rather than exactly on it.
 */
const TIERS: Record<LocalKey, TierRow[]> = {
  zimage: [
    { minGb: 23.5, tier: 'best', note: 'Runs fully resident. The fastest option here.' },
    { minGb: 15.5, tier: 'good', note: 'Fits with int8 weights.' },
    { minGb: 9.5, tier: 'ok', note: 'Runs with CPU offload. Slower first render.' },
    { minGb: 0, tier: 'heavy', note: 'Will run, but expect offload and slow steps.' },
  ],
  flux2: [
    { minGb: 23.5, tier: 'best', note: 'Klein 4B stays resident, and dev is within reach.' },
    { minGb: 15.5, tier: 'good', note: 'Klein 4B fits with int8 and encoder eviction.' },
    { minGb: 11.5, tier: 'ok', note: 'Klein 4B runs quantized with offload.' },
    { minGb: 0, tier: 'heavy', note: 'Will run, but slowly. Try Z-Image first.' },
  ],
  krea2: [
    { minGb: 31.5, tier: 'best', note: 'Full bf16 weights stay resident.' },
    { minGb: 23.5, tier: 'good', note: 'Loads in NF4 or int8.' },
    { minGb: 15.5, tier: 'ok', note: 'NF4 plus offload. Noticeably slower.' },
    {
      minGb: 0,
      tier: 'heavy',
      note: 'Will run, but this is the heaviest model here. Expect long renders.',
    },
  ],
  // Never 'best', and the notes lead with system RAM rather than the card, because that is what
  // actually decides: the 4-bit conditioner alone sits resident at 19.5 GB, and the measured peak
  // on an L40S was 24.8 GB of VRAM against 52.7 GB of RAM. A card that clears the VRAM bar on a
  // 32 GB machine still will not run this.
  minimaxh3: [
    { minGb: 39.5, tier: 'good', note: 'Runs. 64 GB of system RAM matters more than the card.' },
    { minGb: 23.5, tier: 'ok', note: 'Tight: 19.5 GB goes to the conditioner before any frames.' },
    {
      minGb: 0,
      tier: 'heavy',
      note: 'Expect long renders. Needs 64 GB of RAM and 144 GB on disk.',
    },
  ],
  // Measured, not estimated: streaming weights peaked at 7.71 GiB and took 944s cold, while the
  // resident path peaked at 32.19 GiB and got a cached render to 229s. The card decides which of
  // those two experiences you get, so the tiers are drawn at that boundary rather than at "fits".
  ltx25: [
    {
      minGb: 39.5,
      tier: 'good',
      note: 'Holds the transformer resident. Roughly 4x faster than streaming.',
    },
    { minGb: 23.5, tier: 'ok', note: 'Streams weights from RAM. It renders, slowly.' },
    { minGb: 0, tier: 'heavy', note: 'Streams from RAM or disk. Expect several minutes a clip.' },
  ],
  training: [
    { minGb: 23.5, tier: 'best', note: 'Comfortable for LoRA training.' },
    { minGb: 15.5, tier: 'ok', note: 'Trainable at 512 to 768 px with a small batch.' },
    { minGb: 0, tier: 'heavy', note: 'Will run at a small size. Training wants about 16 GB.' },
  ],
}

/** Static requirement copy, shown when the hardware cannot be read. */
const STATIC_NOTE: Record<LocalKey, string> = {
  ltx25: 'Around 48 GB of VRAM to stay resident, and 71 GB on disk. It streams on less.',
  minimaxh3: '64 GB of system RAM and 144 GB on disk. The RAM is the binding constraint.',
  zimage: 'Around 12 GB of VRAM for a comfortable run.',
  flux2: 'Around 16 GB of VRAM for a comfortable run.',
  krea2: 'Around 24 GB of VRAM for a comfortable run.',
  training: 'Around 16 GB of VRAM to train at 512 px.',
}

export interface Advice {
  tier: Tier | null
  note: string
}

export function tierFor(key: StarterKey, reading: VramReading): Advice {
  if (reading.state !== 'known') return { tier: null, note: STATIC_NOTE[key] }
  const row = TIERS[key].find((r) => reading.totalGb >= r.minGb)
  // The last row is minGb 0, so this only falls through if a table is mis-edited.
  return row ? { tier: row.tier, note: row.note } : { tier: null, note: STATIC_NOTE[key] }
}

const SCORE: Record<Tier, number> = { best: 3, good: 2, ok: 1, heavy: 0 }
/** Ties break lightest-first, so a new user is pointed at the quickest thing that works. */
// 'minimaxh3' and 'ltx25' are deliberately absent. Both can score a tier, but the ribbon points
// a new user at their first render, and no first render should begin with a 71 GB download.
const PREFERENCE: LocalKey[] = ['zimage', 'flux2', 'krea2', 'training']

/**
 * The one card that gets the ribbon. Without a hardware reading this is Z-Image, labelled as a
 * neutral starting point rather than as a claim about a GPU we could not see.
 */
export function pickRecommended(reading: VramReading): StarterKey {
  if (reading.state !== 'known') return 'zimage'
  return PREFERENCE.reduce((best, key) => {
    const a = SCORE[tierFor(key, reading).tier ?? 'heavy']
    const b = SCORE[tierFor(best, reading).tier ?? 'heavy']
    return a > b ? key : best
  }, PREFERENCE[0])
}

/** The ribbon's wording, which must not imply hardware knowledge we do not have. */
export function recommendedLabel(reading: VramReading): string {
  return reading.state === 'known' ? 'Recommended for your GPU' : 'Best place to start'
}
