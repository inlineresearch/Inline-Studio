/**
 * Unit tests for resolveInputThumbs - the shared Frame/Generate-node input resolver. Pure (no DB
 * or DOM): given a frame's inputs + the store slices, it maps each input to a displayable
 * thumbnail. Covers asset inputs, a flow link to a source frame's hero take, and the fallback to a
 * source frame's own imported asset when it hasn't generated yet (the same chain the ComfyUI
 * upload path relies on).
 */
import { describe, expect, it } from 'vitest'
import type { Asset, Frame, FrameInput, Take } from '@shared/types'
import { resolveInputThumbs, type InputThumbCtx } from './inputThumbs'

const asset = (over: Partial<Asset>): Asset => ({
  id: 'a1',
  projectId: 'p',
  folderId: null,
  name: 'a.png',
  filePath: 'assets/a.png',
  kind: 'image',
  thumbPath: null,
  previewPath: null,
  createdAt: 0,
  ...over,
})

const frame = (over: Partial<Frame>): Frame => ({
  id: 'f1',
  sequenceId: 's',
  name: '1',
  kind: 'image',
  position: 0,
  inputAssetId: null,
  heroTakeId: null,
  provider: 'unset',
  modelId: null,
  params: {},
  workflowTemplateId: null,
  comfyWorkflowName: null,
  comfyWorkflowReady: false,
  createdAt: 0,
  updatedAt: 0,
  ...over,
})

const take = (over: Partial<Take>): Take => ({
  id: 't1',
  frameId: 'src',
  filePath: 'takes/t1.png',
  kind: 'image',
  params: {},
  comfyPromptId: null,
  createdAt: 0,
  ...over,
})

const input = (over: Partial<FrameInput>): FrameInput => ({
  id: 'i1',
  frameId: 'f1',
  assetId: null,
  sourceFrameId: null,
  position: 0,
  ...over,
})

const emptyCtx: InputThumbCtx = { assets: [], allFrames: [], takesByFrame: {}, inputsByFrame: {} }

describe('resolveInputThumbs', () => {
  it('resolves an asset input to its media, keyed by the input row id', () => {
    const a = asset({ id: 'a1', filePath: 'assets/a.png', kind: 'image' })
    const thumbs = resolveInputThumbs([input({ id: 'i1', assetId: 'a1' })], {
      ...emptyCtx,
      assets: [a],
    })
    expect(thumbs).toHaveLength(1)
    expect(thumbs[0]).toMatchObject({ id: 'i1', assetId: 'a1', kind: 'image' })
    expect(thumbs[0].url).toContain('assets/a.png')
  })

  it("resolves a flow input to the source frame's hero take", () => {
    const src = frame({ id: 'src', heroTakeId: 't1' })
    const thumbs = resolveInputThumbs([input({ id: 'i2', sourceFrameId: 'src' })], {
      ...emptyCtx,
      allFrames: [src],
      takesByFrame: { src: [take({ id: 't1', frameId: 'src', filePath: 'takes/t1.png' })] },
    })
    expect(thumbs).toHaveLength(1)
    expect(thumbs[0]).toMatchObject({ id: 'i2', assetId: null, kind: 'image' })
    expect(thumbs[0].url).toContain('takes/t1.png')
  })

  it("falls back to the source frame's imported asset when it has no take yet", () => {
    const src = frame({ id: 'src' })
    const a = asset({ id: 'srcAsset', filePath: 'assets/src.png' })
    const thumbs = resolveInputThumbs([input({ id: 'i3', sourceFrameId: 'src' })], {
      ...emptyCtx,
      assets: [a],
      allFrames: [src],
      inputsByFrame: { src: [input({ id: 'si', frameId: 'src', assetId: 'srcAsset' })] },
    })
    expect(thumbs).toHaveLength(1)
    expect(thumbs[0].url).toContain('assets/src.png')
  })

  it('drops inputs that resolve to nothing', () => {
    expect(resolveInputThumbs([input({ id: 'x', assetId: 'missing' })], emptyCtx)).toHaveLength(0)
  })
})
