import { describe, it, expect } from 'vitest'
import {
  constantEndpoint,
  approxPrice,
  selectParam,
  numberParam,
  seedParam,
  putSeed,
} from './builders'

describe('constantEndpoint', () => {
  it('always returns the given id', () => {
    const r = constantEndpoint('fal-ai/x')
    expect(r()).toBe('fal-ai/x')
  })
})

describe('approxPrice', () => {
  it('wraps an amount as approximate', () => {
    expect(approxPrice(0.12)).toEqual({ amount: 0.12, approx: true })
  })
})

describe('param builders', () => {
  it('selectParam maps values to options and is advanced by default', () => {
    expect(selectParam('r', 'Res', ['1K', '2K'], '1K')).toEqual({
      key: 'r',
      label: 'Res',
      widget: 'select',
      options: [
        { value: '1K', label: '1K' },
        { value: '2K', label: '2K' },
      ],
      default: '1K',
      advanced: true,
    })
  })

  it('numberParam carries range', () => {
    expect(numberParam('n', 'Count', 1, { min: 1, max: 4, step: 1 })).toMatchObject({
      widget: 'number',
      default: 1,
      min: 1,
      max: 4,
      step: 1,
    })
  })

  it('seedParam is a -1 default number', () => {
    expect(seedParam()).toMatchObject({ key: 'seed', widget: 'number', default: -1 })
  })
})

describe('putSeed', () => {
  it('adds seed only when finite and >= 0', () => {
    const a: Record<string, unknown> = {}
    putSeed(a, { seed: 42 })
    expect(a.seed).toBe(42)
    const b: Record<string, unknown> = {}
    putSeed(b, { seed: -1 })
    expect(b.seed).toBeUndefined()
    const c: Record<string, unknown> = {}
    putSeed(c, {})
    expect(c.seed).toBeUndefined()
  })
})
