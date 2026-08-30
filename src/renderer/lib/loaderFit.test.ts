import { describe, expect, it } from 'vitest'
import {
  LOADER_CHROME_H,
  LOADER_LONG_EDGE,
  LOADER_MAX_BODY,
  LOADER_MAX_W,
  LOADER_MIN_BODY,
  LOADER_MIN_W,
  fitLoaderSize,
  needsBlurFill,
} from './loaderFit'

const body = (h: number): number => h - LOADER_CHROME_H

describe('fitLoaderSize', () => {
  it('matches the long edge on both orientations', () => {
    expect(fitLoaderSize(2 / 3)).toEqual({ width: 227, height: 341 + LOADER_CHROME_H })
    expect(fitLoaderSize(16 / 9)).toEqual({ width: 340, height: 191 + LOADER_CHROME_H })
    expect(fitLoaderSize(1)).toEqual({ width: 340, height: 340 + LOADER_CHROME_H })
  })

  it('reproduces the media aspect within a pixel when nothing clamps', () => {
    for (const aspect of [0.75, 1, 1.25, 1.5, 16 / 9]) {
      const { width, height } = fitLoaderSize(aspect)
      expect(Math.abs(width / body(height) - aspect)).toBeLessThan(0.01)
    }
  })

  it('scales a narrow portrait up to the minimum width without introducing bars', () => {
    const { width, height } = fitLoaderSize(9 / 16)
    expect(width).toBe(LOADER_MIN_W)
    // The body is derived from the clamped width, so the aspect survives the floor.
    expect(Math.abs(width / body(height) - 9 / 16)).toBeLessThan(0.01)
  })

  it('leaves bars only where the body itself clamps', () => {
    const wide = fitLoaderSize(3)
    expect(body(wide.height)).toBe(LOADER_MIN_BODY)
    expect(wide.width / body(wide.height)).toBeLessThan(3)

    const tall = fitLoaderSize(0.3)
    expect(body(tall.height)).toBe(LOADER_MAX_BODY)
    expect(tall.width / body(tall.height)).toBeGreaterThan(0.3)
  })

  it('stays inside every bound for any aspect', () => {
    for (let a = 0.05; a <= 20; a += 0.05) {
      const { width, height } = fitLoaderSize(a)
      expect(width).toBeGreaterThanOrEqual(LOADER_MIN_W)
      expect(width).toBeLessThanOrEqual(LOADER_MAX_W)
      expect(body(height)).toBeGreaterThanOrEqual(LOADER_MIN_BODY)
      expect(body(height)).toBeLessThanOrEqual(LOADER_MAX_BODY)
    }
  })

  it('keeps an unusable aspect out of the stored size', () => {
    for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      const { width, height } = fitLoaderSize(bad)
      expect(Number.isFinite(width) && Number.isFinite(height)).toBe(true)
      expect(width).toBe(LOADER_LONG_EDGE)
      expect(body(height)).toBe(LOADER_LONG_EDGE)
    }
  })
})

describe('needsBlurFill', () => {
  it('is quiet on a node still at its fitted size', () => {
    for (const aspect of [2 / 3, 1, 16 / 9, 9 / 16]) {
      const { width, height } = fitLoaderSize(aspect)
      expect(needsBlurFill(aspect, width / body(height))).toBe(false)
    }
  })

  it('fires once the box no longer matches the media', () => {
    expect(needsBlurFill(2 / 3, 16 / 9)).toBe(true)
    expect(needsBlurFill(1, 1.5)).toBe(true)
  })

  it('fires on the aspects whose body clamps, since those keep bars by design', () => {
    for (const aspect of [3, 0.3]) {
      const { width, height } = fitLoaderSize(aspect)
      expect(needsBlurFill(aspect, width / body(height))).toBe(true)
    }
  })

  it('is quiet on a degenerate box', () => {
    expect(needsBlurFill(1, 0)).toBe(false)
    expect(needsBlurFill(Number.NaN, 1)).toBe(false)
  })
})
