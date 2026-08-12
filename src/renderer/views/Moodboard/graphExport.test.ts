import { beforeEach, describe, expect, it } from 'vitest'
import type { InlineStudioApi } from '@shared/ipc'
import { ok } from '@shared/result'
import type { MoodboardConnector, MoodboardItem } from '@shared/types'
import { parseRecipeJson } from '../../lib/pngRecipe'
import { setStudioClient } from '../../lib/studio'
import { useMoodboardStore } from '../../store/moodboardStore'
import { duplicateGraph, graphJson, graphRecipe, graphSlice, unsupportedTypes } from './graphExport'

function item(id: string, type: string, data: Record<string, unknown> = {}): MoodboardItem {
  return {
    id,
    projectId: 'p',
    surface: 'studio',
    type,
    assetId: null,
    frameId: null,
    parentId: null,
    data,
    x: 0,
    y: 0,
    width: 100,
    height: 100,
    rotation: 0,
    zIndex: 0,
    createdAt: 0,
    updatedAt: 0,
  } as MoodboardItem
}

function connector(from: string, to: string): MoodboardConnector {
  return {
    id: `${from}-${to}`,
    projectId: 'p',
    surface: 'studio',
    fromItemId: from,
    toItemId: to,
    label: null,
    data: { sourceHandle: 'out', targetHandle: 'prompt' },
    createdAt: 0,
  } as MoodboardConnector
}

const PROMPT = item('prompt1', 'prompt', { promptText: 'a neon city' })
const CORE = item('core1', 'core', {
  core: { type: 'alibaba/z-image-turbo', params: { steps: 8 }, output: { takeId: 'tk1' } },
})
const STRAY = item('core2', 'core', { core: { type: 'other' } })

beforeEach(() => {
  useMoodboardStore.setState({
    items: [PROMPT, CORE, STRAY],
    connectors: [connector('prompt1', 'core1')],
  })
})

describe('graphSlice', () => {
  it('takes only the connected graph, not the whole board', () => {
    const slice = graphSlice('core1')
    expect(slice.items.map((i) => i.id).sort()).toEqual(['core1', 'prompt1'])
    expect(slice.connectors).toHaveLength(1)
  })

  it('an unconnected node is its own one-node graph', () => {
    expect(graphSlice('core2').items.map((i) => i.id)).toEqual(['core2'])
    expect(graphSlice('core2').connectors).toEqual([])
  })
})

describe('graphRecipe', () => {
  it('emits the recipe shape the PNG importer already reads', () => {
    const recipe = graphRecipe('core1')
    expect(recipe.app).toBe('inline-studio')
    expect(recipe.version).toBe(1)
    expect(recipe.target).toBe('core1')
    expect(recipe.coreType).toBe('alibaba/z-image-turbo')
    expect(recipe.params).toEqual({ steps: 8 })
    expect(recipe.prompt).toBe('a neon city')
    expect(recipe.graph?.items).toHaveLength(2)
    expect(recipe.graph?.connectors).toEqual([
      {
        fromItemId: 'prompt1',
        toItemId: 'core1',
        data: { sourceHandle: 'out', targetHandle: 'prompt' },
      },
    ])
  })

  it('keeps the take history out of the exported graph', () => {
    // A recipe describes how to make the image, not the images already made.
    const core = graphRecipe('core1').graph?.items.find((i) => i.id === 'core1')
    expect((core?.data.core as Record<string, unknown>).output).toBeUndefined()
  })

  it('round-trips through JSON unchanged', () => {
    const recipe = graphRecipe('core1')
    expect(JSON.parse(JSON.stringify(recipe))).toEqual(recipe)
  })

  it('exported JSON is accepted by the drop-side parser', () => {
    // Export then drop is the whole point: the file has to satisfy the same `app` guard the PNG
    // chunk does, or dragging it back onto the canvas silently does nothing.
    const parsed = parseRecipeJson(graphJson('core1'))
    expect(parsed).not.toBeNull()
    expect(parsed?.target).toBe('core1')
    expect(parsed?.graph?.items).toHaveLength(2)
    expect(parsed?.graph?.connectors).toHaveLength(1)
  })
})

describe('unsupportedTypes', () => {
  it('is empty for a graph the importer can rebuild', () => {
    expect(unsupportedTypes('core1')).toEqual([])
  })

  it('names the node types a re-import would drop', () => {
    const director = item('dir1', 'director')
    useMoodboardStore.setState({
      items: [CORE, director],
      connectors: [connector('core1', 'dir1')],
    })
    expect(unsupportedTypes('core1')).toEqual(['director'])
  })
})

describe('duplicateGraph', () => {
  let created: MoodboardItem[]
  let connectorCalls: { from: string; to: string; sh: string | null; th: string | null }[]

  beforeEach(() => {
    created = []
    connectorCalls = []
    let n = 0
    const make = (type: string, x: number, y: number): MoodboardItem => {
      n += 1
      const copy = { ...item(`new${n}`, type), x, y }
      created.push(copy)
      return copy
    }
    setStudioClient({
      moodboard: {
        addCoreNode: async (_type: string, x: number, y: number) => ok(make('core', x, y)),
        addPrompt: async (x: number, y: number) => ok(make('prompt', x, y)),
        updateItem: async (id: string, patch: Record<string, unknown>) => {
          const target = created.find((c) => c.id === id)
          return ok({ ...(target as MoodboardItem), ...patch })
        },
        createConnector: async (from: string, to: string, sh: string | null, th: string | null) => {
          connectorCalls.push({ from, to, sh, th })
          return ok({ ...connector(from, to), id: `c${connectorCalls.length}` })
        },
      },
    } as unknown as InlineStudioApi)
  })

  it('copies a core + prompt graph and re-creates its wiring', async () => {
    // copyOne used to fall through to `default: return null` for core/prompt, so this whole
    // action silently did nothing on a generation graph.
    const count = await duplicateGraph('core1')

    expect(count).toBe(2)
    expect(created.map((c) => c.type).sort()).toEqual(['core', 'prompt'])
    expect(connectorCalls).toHaveLength(1)
    expect(connectorCalls[0].sh).toBe('out')
    expect(connectorCalls[0].th).toBe('prompt')
    // Wired between the copies, not back to the originals.
    expect(connectorCalls[0].from.startsWith('new')).toBe(true)
    expect(connectorCalls[0].to.startsWith('new')).toBe(true)
  })

  it('offsets the copies so they do not land on top of the originals', async () => {
    await duplicateGraph('core1', 48)
    expect(created.every((c) => c.x === 48 && c.y === 48)).toBe(true)
  })

  it('does not carry the original take history onto the copy', async () => {
    await duplicateGraph('core1')
    const copies = useMoodboardStore.getState().items.filter((i) => i.id.startsWith('new'))
    const copy = copies.find((i) => (i.data as { core?: unknown }).core)
    const core = (copy?.data as { core?: Record<string, unknown> } | undefined)?.core
    expect(core?.type).toBe('alibaba/z-image-turbo')
    expect(core?.params).toEqual({ steps: 8 })
    // The copy is a fresh slot, so it must not claim the original's renders.
    expect(core?.output).toBeUndefined()
  })
})
