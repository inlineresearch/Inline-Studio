import { describe, expect, it } from 'vitest'
import type { ClipGrid } from './clipGrid'
import {
  CLIP_GRIDS,
  H3_MIN_CLIP_FRAMES,
  minClipFrames,
  resolveClipLength,
  resolveClipLengthFor,
  snapClipFrames,
  snapClipFramesFor,
} from './clipGrid'

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

describe('LTX-2.5 clip grid', () => {
  const grid = CLIP_GRIDS['ltx-2-5'] as ClipGrid

  it('snaps down onto 8n+1', () => {
    expect(snapClipFramesFor(grid, 121)).toBe(121)
    expect(snapClipFramesFor(grid, 120)).toBe(113)
    expect(snapClipFramesFor(grid, 100)).toBe(97)
  })

  it('floors at one chunk plus the head rather than refusing', () => {
    expect(minClipFrames(grid)).toBe(9)
    expect(snapClipFramesFor(grid, 1)).toBe(9)
    expect(snapClipFramesFor(grid, 8)).toBe(9)
  })

  it('resolves seconds the way the panel shows them', () => {
    expect(resolveClipLengthFor(grid, 2)).toEqual({ frames: 41, seconds: 41 / 24 })
    expect(resolveClipLengthFor(grid, 5)).toEqual({ frames: 113, seconds: 113 / 24 })
  })

  it('never rounds up past what the clip holds', () => {
    for (let frames = 9; frames < 400; frames += 1) {
      expect(snapClipFramesFor(grid, frames)).toBeLessThanOrEqual(frames)
    }
  })
})

describe('clip grid coverage', () => {
  // Exhaustive by construction (CLIP_GRIDS is a Record over TrainingArch), so this guards the
  // values rather than the keys: a grid whose offset exceeds its stride would snap upward.
  it('every declared grid is a valid stride and offset', () => {
    for (const [arch, grid] of Object.entries(CLIP_GRIDS)) {
      if (!grid) continue
      expect(grid.grid, arch).toBeGreaterThan(0)
      expect(grid.offset, arch).toBeGreaterThanOrEqual(0)
      expect(grid.fps, arch).toBeGreaterThan(0)
    }
  })
})
