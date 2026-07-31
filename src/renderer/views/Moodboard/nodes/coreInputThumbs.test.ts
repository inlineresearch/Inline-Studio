import { describe, expect, it } from 'vitest'

import { resolveCoreInputThumbs, type ThumbContext } from './coreInputThumbs'
import type { Asset, Frame, MoodboardConnector, MoodboardItem, Take } from '@shared/types'

const item = (over: Partial<MoodboardItem> & { id: string; type: MoodboardItem['type'] }) =>
  ({ projectId: 'p', surface: 'studio', x: 0, y: 0, zIndex: 0, data: {}, ...over }) as MoodboardItem

const asset = (id: string): Asset =>
  ({ id, projectId: 'p', name: id, filePath: `assets/${id}.png`, kind: 'image' }) as Asset

const wire = (from: string, to: string, handle: string): MoodboardConnector => ({
  id: `${from}->${to}:${handle}`,
  projectId: 'p',
  surface: 'studio',
  fromItemId: from,
  toItemId: to,
  label: null,
  createdAt: 0,
  data: { targetHandle: handle },
})

const ctx = (over: Partial<ThumbContext>): ThumbContext => ({
  items: [],
  connectors: [],
  assets: [],
  frames: [],
  takesByFrame: {},
  ...over,
})

describe('resolveCoreInputThumbs', () => {
  it('numbers references from 1 in wiring order', () => {
    const thumbs = resolveCoreInputThumbs(
      'gen',
      'image',
      ctx({
        items: [
          item({ id: 'gen', type: 'core' }),
          item({ id: 'a', type: 'asset', assetId: 'a1' }),
          item({ id: 'b', type: 'asset', assetId: 'a2' }),
        ],
        connectors: [wire('a', 'gen', 'image'), wire('b', 'gen', 'image')],
        assets: [asset('a1'), asset('a2')],
      }),
    )
    expect(thumbs.map((t) => [t.index, t.filePath])).toEqual([
      [1, 'assets/a1.png'],
      [2, 'assets/a2.png'],
    ])
  })

  it('ignores wires into other handles', () => {
    const thumbs = resolveCoreInputThumbs(
      'gen',
      'image',
      ctx({
        items: [item({ id: 'gen', type: 'core' }), item({ id: 'p', type: 'asset', assetId: 'a1' })],
        connectors: [wire('p', 'gen', 'control_image')],
        assets: [asset('a1')],
      }),
    )
    expect(thumbs).toEqual([])
  })

  it('expands a Load Assets node into one numbered reference per asset', () => {
    // Must match the engine's fan-out, or the numbers on the card would not be the numbers the
    // prompt addresses.
    const thumbs = resolveCoreInputThumbs(
      'gen',
      'image',
      ctx({
        items: [
          item({ id: 'gen', type: 'core' }),
          item({ id: 'loader', type: 'loader', data: { assetIds: ['a1', 'a2', 'a3'] } }),
        ],
        connectors: [wire('loader', 'gen', 'image')],
        assets: [asset('a1'), asset('a2'), asset('a3')],
      }),
    )
    expect(thumbs.map((t) => t.index)).toEqual([1, 2, 3])
    expect(thumbs.map((t) => t.filePath)).toEqual([
      'assets/a1.png',
      'assets/a2.png',
      'assets/a3.png',
    ])
  })

  it('keeps numbering continuous across a loader and a plain asset', () => {
    const thumbs = resolveCoreInputThumbs(
      'gen',
      'image',
      ctx({
        items: [
          item({ id: 'gen', type: 'core' }),
          item({ id: 'loader', type: 'loader', data: { assetIds: ['a1', 'a2'] } }),
          item({ id: 'solo', type: 'asset', assetId: 'a3' }),
        ],
        connectors: [wire('loader', 'gen', 'image'), wire('solo', 'gen', 'image')],
        assets: [asset('a1'), asset('a2'), asset('a3')],
      }),
    )
    expect(thumbs.map((t) => [t.index, t.filePath])).toEqual([
      [1, 'assets/a1.png'],
      [2, 'assets/a2.png'],
      [3, 'assets/a3.png'],
    ])
  })

  it('takes a frame at its hero take, not the newest', () => {
    const takes = [
      { id: 't1', frameId: 'f1', filePath: 'takes/1.png' } as Take,
      { id: 't2', frameId: 'f1', filePath: 'takes/2.png' } as Take,
    ]
    const thumbs = resolveCoreInputThumbs(
      'gen',
      'image',
      ctx({
        items: [item({ id: 'gen', type: 'core' }), item({ id: 'f', type: 'frame', frameId: 'f1' })],
        connectors: [wire('f', 'gen', 'image')],
        frames: [{ id: 'f1', heroTakeId: 't1' } as Frame],
        takesByFrame: { f1: takes },
      }),
    )
    expect(thumbs[0]?.filePath).toBe('takes/1.png')
  })

  it('takes a wired Core node at its current output', () => {
    const thumbs = resolveCoreInputThumbs(
      'gen',
      'image',
      ctx({
        items: [
          item({ id: 'gen', type: 'core' }),
          item({
            id: 'up',
            type: 'core',
            data: {
              core: {
                type: 'black-forest-labs/flux-2',
                params: {},
                output: { takeId: 't', filePath: 'takes/up.png', kind: 'image' },
              },
            },
          }),
        ],
        connectors: [wire('up', 'gen', 'image')],
      }),
    )
    expect(thumbs[0]?.filePath).toBe('takes/up.png')
  })

  it('skips sources with nothing rendered or nothing selected', () => {
    const thumbs = resolveCoreInputThumbs(
      'gen',
      'image',
      ctx({
        items: [
          item({ id: 'gen', type: 'core' }),
          item({ id: 'empty', type: 'controlSpace' }),
          item({ id: 'unrendered', type: 'frame', frameId: 'f9' }),
          item({ id: 'ok', type: 'asset', assetId: 'a1' }),
        ],
        connectors: [
          wire('empty', 'gen', 'image'),
          wire('unrendered', 'gen', 'image'),
          wire('ok', 'gen', 'image'),
        ],
        assets: [asset('a1')],
      }),
    )
    // The one resolvable source is reference 1, so the numbering never has a gap.
    expect(thumbs).toHaveLength(1)
    expect(thumbs[0]).toMatchObject({ index: 1, filePath: 'assets/a1.png' })
  })
})
