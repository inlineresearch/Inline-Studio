import { describe, expect, it } from 'vitest'
import { H3_MIN_CLIP_FRAMES, resolveClipLength, snapClipFrames } from './clipGrid'

describe('H3 clip grid', () => {
  // Pinned against Core's trim_reference_num_frames, which is the authority. If these drift, the
  // Trainer shows a duration the run will not honour.
  it('snaps down onto 17n+5, never up', () => {
    expect([5, 22, 39, 56, 73, 90, 107, 124, 141].map(snapClipFrames)).toEqual([
      22, 22, 39, 56, 73, 90, 107, 124, 141,
    ])
    expect(snapClipFrames(120)).toBe(107)
    expect(snapClipFrames(23)).toBe(22)
  })

  it('never returns less than the floor', () => {
    expect(snapClipFrames(1)).toBe(H3_MIN_CLIP_FRAMES)
    expect(snapClipFrames(0)).toBe(H3_MIN_CLIP_FRAMES)
    expect(snapClipFrames(Number.NaN)).toBe(H3_MIN_CLIP_FRAMES)
  })

  it('resolves the seconds a user typed to what actually trains', () => {
    // The case that prompted this: 5s of source trains on 4.458s, losing the last half second.
    expect(resolveClipLength(5)).toEqual({ frames: 107, seconds: 107 / 24 })
    expect(resolveClipLength(1)).toEqual({ frames: 22, seconds: 22 / 24 })
    expect(resolveClipLength(5.2)).toEqual({ frames: 124, seconds: 124 / 24 })
  })
})
