import { beforeEach, describe, expect, it } from 'vitest'
import type { Asset, MoodboardItem } from '@shared/types'
import { useAssetStore } from '../store/assetStore'
import { useMoodboardStore } from '../store/moodboardStore'
import type { Recipe } from './pngRecipe'
import { buildGraphFromRecipe } from './recipeGraph'

interface Wire {
  from: string
  to: string
  sourceHandle: string | null
  targetHandle: string | null
}

let created: { id: string; type: string }[] = []
let wires: Wire[] = []
let patched: Record<string, Record<string, unknown>> = {}

function stub(type: string): MoodboardItem {
  const id = `new-${created.length}`
  created.push({ id, type })
  return { id, type, data: {} } as MoodboardItem
}

beforeEach(() => {
  created = []
  wires = []
  patched = {}
  useAssetStore.setState({ assets: [{ id: 'here' } as Asset] })
  useMoodboardStore.setState({
    addCoreNode: async () => stub('core'),
    addLoader: async () => stub('loader'),
    addPrompt: async () => stub('prompt'),
    updateItem: async (id, patch) => {
      patched[id] = (patch.data ?? {}) as Record<string, unknown>
    },
    connect: async (from, to, sourceHandle, targetHandle) => {
      wires.push({
        from,
        to,
        sourceHandle: sourceHandle ?? null,
        targetHandle: targetHandle ?? null,
      })
    },
  })
})

/** A gen node fed by one media source, the shape an exported graph takes. */
function fedBy(source: { id: string; type: string; data?: Record<string, unknown> }): Recipe {
  return {
    version: 1,
    app: 'inline-studio',
    target: 'core1',
    graph: {
      items: [
        { id: 'core1', type: 'core', data: { core: { type: 'z', params: {} } }, x: 0, y: 0 },
        { id: source.id, type: source.type, data: source.data ?? {}, x: 0, y: 0 },
      ],
      connectors: [
        {
          fromItemId: source.id,
          toItemId: 'core1',
          data: { sourceHandle: 'image', targetHandle: 'image' },
        },
      ],
    },
  } as unknown as Recipe
}

describe('buildGraphFromRecipe', () => {
  it('stands an empty loader in for an asset node so the wire survives the import', async () => {
    await buildGraphFromRecipe(fedBy({ id: 'a1', type: 'asset' }), { x: 0, y: 0 })

    expect(created.map((c) => c.type)).toEqual(['core', 'loader'])
    expect(wires).toHaveLength(1)
    expect(wires[0].targetHandle).toBe('image')
    // The loader's own output handle, not the one the asset node named.
    expect(wires[0].sourceHandle).toBeNull()
  })

  it('drops loader assets this project has never seen, and keeps the ones it has', async () => {
    const loader = { id: 'l1', type: 'loader', data: { assetIds: ['here', 'from-elsewhere'] } }
    await buildGraphFromRecipe(fedBy(loader), { x: 0, y: 0 })

    const loaderId = created.find((c) => c.type === 'loader')?.id ?? ''
    expect(patched[loaderId]?.assetIds).toEqual(['here'])
  })

  it('leaves a loader with nothing to keep empty rather than pointing it at dead ids', async () => {
    const loader = { id: 'l1', type: 'loader', data: { assetIds: ['from-elsewhere'] } }
    await buildGraphFromRecipe(fedBy(loader), { x: 0, y: 0 })

    const loaderId = created.find((c) => c.type === 'loader')?.id ?? ''
    expect(patched[loaderId]).toBeUndefined()
    expect(wires).toHaveLength(1)
  })
})
