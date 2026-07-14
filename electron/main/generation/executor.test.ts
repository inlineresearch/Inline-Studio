import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock the side-effecting deps so the executor runs without a DB, network, or filesystem.
vi.mock('../db', () => ({
  getOpenProjectFolder: vi.fn(() => '/proj'),
  getDb: vi.fn(),
}))
vi.mock('../frames/store', () => ({
  getFalFrame: vi.fn(),
  frameInputAssetPaths: vi.fn(() => []),
  addTake: vi.fn(),
  setHero: vi.fn(),
}))
vi.mock('../fal/client', () => ({
  fileToInputUrl: vi.fn(async (p: string) => `data:${p}`),
  submit: vi.fn(),
  submitRequest: vi.fn(),
  pollResult: vi.fn(),
  cancelRequest: vi.fn(),
}))
vi.mock('./pending', () => ({
  recordPending: vi.fn(),
  deletePendingByFrame: vi.fn(),
  listPending: vi.fn(() => []),
}))
vi.mock('../moodboard/store', () => ({
  promptTextForFrame: vi.fn(() => 'a test prompt'),
}))
vi.mock('node:fs', () => ({
  writeFileSync: vi.fn(),
  mkdirSync: vi.fn(),
}))

import {
  runGraph,
  resumePendingGenerations,
  upstreamClosure,
  topoSort,
  type GenEmitter,
  type EdgesFn,
} from './executor'
import * as framesStore from '../frames/store'
import * as falClient from '../fal/client'
import * as pendingStore from './pending'
import * as moodboardStore from '../moodboard/store'

const getFalFrame = framesStore.getFalFrame as unknown as ReturnType<typeof vi.fn>
const frameInputAssetPaths = framesStore.frameInputAssetPaths as unknown as ReturnType<typeof vi.fn>
const addTake = framesStore.addTake as unknown as ReturnType<typeof vi.fn>
const setHero = framesStore.setHero as unknown as ReturnType<typeof vi.fn>
const submit = falClient.submit as unknown as ReturnType<typeof vi.fn>
const pollResult = falClient.pollResult as unknown as ReturnType<typeof vi.fn>
const listPending = pendingStore.listPending as unknown as ReturnType<typeof vi.fn>
const promptTextForFrame = moodboardStore.promptTextForFrame as unknown as ReturnType<typeof vi.fn>

/** An emitter that records the calls it receives, in order. */
function recordingEmitter(): { emit: GenEmitter; events: string[] } {
  const events: string[] = []
  return {
    events,
    emit: {
      progress: (frameId) => events.push(`progress:${frameId}`),
      nodeDone: (frameId, take) => events.push(`nodeDone:${frameId}:${take.id}`),
      done: (target) => events.push(`done:${target}`),
      error: (target, error, frameId) => events.push(`error:${target}:${frameId ?? ''}:${error}`),
    },
  }
}

const edgesFrom =
  (adj: Record<string, string[]>): EdgesFn =>
  (id) =>
    adj[id] ?? []

beforeEach(() => {
  vi.clearAllMocks()
  frameInputAssetPaths.mockReturnValue([])
  promptTextForFrame.mockReturnValue('a test prompt')
  addTake.mockImplementation((input: { frameId: string }) => ({
    id: `take-${input.frameId}`,
    frameId: input.frameId,
    filePath: 'takes/x.png',
    kind: 'image',
    params: {},
    comfyPromptId: null,
    createdAt: 0,
  }))
  // Stub the download step (fetch → arrayBuffer).
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({ ok: true, arrayBuffer: async () => new ArrayBuffer(8) })),
  )
})

describe('upstreamClosure', () => {
  it('collects a linear chain', () => {
    const edges = edgesFrom({ C: ['B'], B: ['A'], A: [] })
    expect(new Set(upstreamClosure('C', edges))).toEqual(new Set(['A', 'B', 'C']))
  })

  it('visits a shared upstream (diamond) once', () => {
    const edges = edgesFrom({ D: ['B', 'C'], B: ['A'], C: ['A'], A: [] })
    const closure = upstreamClosure('D', edges)
    expect(closure.filter((id) => id === 'A')).toHaveLength(1)
    expect(new Set(closure)).toEqual(new Set(['A', 'B', 'C', 'D']))
  })

  it('terminates on a cycle', () => {
    const edges = edgesFrom({ A: ['B'], B: ['A'] })
    expect(new Set(upstreamClosure('A', edges))).toEqual(new Set(['A', 'B']))
  })
})

describe('topoSort', () => {
  it('orders dependencies before dependents (diamond)', () => {
    const edges = edgesFrom({ D: ['B', 'C'], B: ['A'], C: ['A'], A: [] })
    const order = topoSort(['A', 'B', 'C', 'D'], edges)
    expect(order.indexOf('A')).toBeLessThan(order.indexOf('B'))
    expect(order.indexOf('A')).toBeLessThan(order.indexOf('C'))
    expect(order.indexOf('B')).toBeLessThan(order.indexOf('D'))
    expect(order.indexOf('C')).toBeLessThan(order.indexOf('D'))
  })

  it('throws on a cycle', () => {
    const edges = edgesFrom({ A: ['B'], B: ['A'] })
    expect(() => topoSort(['A', 'B'], edges)).toThrow(/cycle/i)
  })
})

describe('runGraph', () => {
  it('runs a single fal node: submits, saves a take, sets hero, emits done', async () => {
    getFalFrame.mockReturnValue({ provider: 'fal', modelId: 'openai/gpt-image-2', params: {} })
    submit.mockResolvedValue({
      requestId: 'r1',
      response: { images: [{ url: 'https://fal/o.png', content_type: 'image/png' }] },
    })
    const { emit, events } = recordingEmitter()

    await runGraph('A', emit)

    expect(submit).toHaveBeenCalledTimes(1)
    expect(submit.mock.calls[0][0]).toBe('openai/gpt-image-2') // text-to-image (no inputs)
    expect(addTake).toHaveBeenCalledTimes(1)
    expect(setHero).toHaveBeenCalledWith('A', 'take-A')
    expect(events).toContain('nodeDone:A:take-A')
    expect(events).toContain('done:A')
    expect(events.some((e) => e.startsWith('error:'))).toBe(false)
  })

  it('resolves an already-produced upstream output as an image input (image-to-image)', async () => {
    // B (fal) has an image input already resolved to an upstream node's take on disk.
    getFalFrame.mockReturnValue({ provider: 'fal', modelId: 'openai/gpt-image-2', params: {} })
    frameInputAssetPaths.mockReturnValue([{ filePath: 'takes/a.png', name: 'a.png' }])
    submit.mockResolvedValue({
      requestId: 'r2',
      response: { images: [{ url: 'https://fal/b.png', content_type: 'image/png' }] },
    })
    const { emit, events } = recordingEmitter()

    await runGraph('B', emit)

    // Only the selected node runs; its image was uploaded and passed via image_urls (editing mode
    // on the same base endpoint — gpt-image-2 has no /image-to-image sub-path).
    expect(submit).toHaveBeenCalledTimes(1)
    expect(falClient.fileToInputUrl).toHaveBeenCalledWith('/proj/takes/a.png')
    expect(submit.mock.calls[0][0]).toBe('openai/gpt-image-2')
    expect(submit.mock.calls[0][1]).toMatchObject({ image_urls: ['data:/proj/takes/a.png'] })
    expect(events).toContain('done:B')
  })

  it('errors when no prompt is connected', async () => {
    getFalFrame.mockReturnValue({ provider: 'fal', modelId: 'openai/gpt-image-2', params: {} })
    promptTextForFrame.mockReturnValue(null)
    const { emit, events } = recordingEmitter()

    await runGraph('A', emit)

    expect(submit).not.toHaveBeenCalled()
    expect(events.some((e) => e.startsWith('error:A:A:'))).toBe(true)
  })

  it('runs a prompt-optional model (Sonilo) without a connected prompt', async () => {
    getFalFrame.mockReturnValue({
      provider: 'fal',
      modelId: 'sonilo/v1.1/video-to-music',
      params: {},
    })
    promptTextForFrame.mockReturnValue(null)
    frameInputAssetPaths.mockReturnValue([{ filePath: 'takes/cut.mp4', name: 'cut.mp4' }])
    submit.mockResolvedValue({
      requestId: 'r3',
      response: { audios: [{ url: 'https://fal/a.m4a', content_type: 'audio/mp4' }] },
    })
    const { emit, events } = recordingEmitter()

    await runGraph('M', emit)

    // The video uploads as video_url; the empty prompt is omitted from the request entirely.
    expect(submit).toHaveBeenCalledTimes(1)
    expect(submit.mock.calls[0][0]).toBe('sonilo/v1.1/video-to-music')
    expect(submit.mock.calls[0][1]).toMatchObject({ video_url: 'data:/proj/takes/cut.mp4' })
    expect(submit.mock.calls[0][1]).not.toHaveProperty('prompt')
    expect(events).toContain('done:M')
    expect(events.some((e) => e.startsWith('error:'))).toBe(false)
  })

  it('emits an error and aborts when a required input is missing', async () => {
    // LTX requires an image; none wired.
    getFalFrame.mockReturnValue({
      provider: 'fal',
      modelId: 'fal-ai/ltx-2.3-quality/image-to-video',
      params: {},
    })
    frameInputAssetPaths.mockReturnValue([])
    const { emit, events } = recordingEmitter()

    await runGraph('V', emit)

    expect(submit).not.toHaveBeenCalled()
    expect(events.some((e) => e.startsWith('error:V:V:'))).toBe(true)
    expect(events).not.toContain('done:V')
  })
})

describe('resumePendingGenerations', () => {
  it('re-polls a persisted run, saves its take, and emits done', async () => {
    listPending.mockReturnValue([
      {
        id: 'p1',
        frameId: 'A',
        modelId: 'openai/gpt-image-2',
        endpoint: 'openai/gpt-image-2',
        requestId: 'r1',
        statusUrl: 'https://q/status',
        responseUrl: 'https://q/response',
        params: {},
        createdAt: 0,
      },
    ])
    pollResult.mockResolvedValue({
      images: [{ url: 'https://fal/o.png', content_type: 'image/png' }],
    })
    const { emit, events } = recordingEmitter()

    await resumePendingGenerations(emit)
    // resumeOne runs fire-and-forget; let its promise chain settle.
    await new Promise((r) => setTimeout(r, 0))

    expect(pollResult).toHaveBeenCalledTimes(1)
    expect(addTake).toHaveBeenCalledTimes(1)
    expect(setHero).toHaveBeenCalledWith('A', 'take-A')
    expect(events).toContain('done:A')
    expect(events.some((e) => e.startsWith('error:'))).toBe(false)
  })

  it('drops a pending run whose model is no longer known', async () => {
    listPending.mockReturnValue([
      {
        id: 'p2',
        frameId: 'B',
        modelId: 'ghost/model',
        endpoint: 'ghost/model',
        requestId: 'r2',
        statusUrl: 's',
        responseUrl: 'r',
        params: {},
        createdAt: 0,
      },
    ])
    const { emit } = recordingEmitter()

    await resumePendingGenerations(emit)

    expect(pollResult).not.toHaveBeenCalled()
    expect(pendingStore.deletePendingByFrame).toHaveBeenCalledWith('B')
  })
})
