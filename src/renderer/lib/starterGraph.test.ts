import { beforeEach, describe, expect, it, vi } from 'vitest'

import { buildStarterGraph } from './starterGraph'
import { recipeFor } from './starterRecipes'
import { useMoodboardStore } from '../store/moodboardStore'
import { useGenerationStore } from '../store/generationStore'
import type { MoodboardItem } from '@shared/types'

const item = (id: string, data: Record<string, unknown> = {}): MoodboardItem =>
  ({ id, type: 'core', x: 0, y: 0, width: 200, height: 120, data }) as unknown as MoodboardItem

type Store = ReturnType<typeof useMoodboardStore.getState>

function stub(over: Partial<Store> = {}) {
  const addCoreNode = vi.fn<Store['addCoreNode']>(async () => item('gen', { keepMe: 1 }))
  const addPrompt = vi.fn<Store['addPrompt']>(async () => item('prompt', { keepMe: 2 }))
  const updateItem = vi.fn<Store['updateItem']>(async () => undefined)
  const connect = vi.fn<Store['connect']>(async () => undefined)
  useMoodboardStore.setState({ addCoreNode, addPrompt, updateItem, connect, ...over })
  return { addCoreNode, addPrompt, updateItem, connect }
}

const ZIMAGE = recipeFor('zimage')!

describe('buildStarterGraph', () => {
  beforeEach(() => {
    useGenerationStore.setState({ error: null })
  })

  it('creates both nodes and wires them, returning [prompt, gen]', async () => {
    const s = stub()
    const ids = await buildStarterGraph(ZIMAGE, { x: 0, y: 0 })
    expect(ids).toEqual(['prompt', 'gen'])
    expect(s.addCoreNode).toHaveBeenCalledWith(
      'alibaba/z-image-turbo',
      expect.any(Number),
      expect.any(Number),
    )
    expect(s.connect).toHaveBeenCalledWith('prompt', 'gen', 'out', 'prompt')
  })

  it('spreads the created data instead of replacing it', async () => {
    // updateItem replaces `data` wholesale (there is no patchData), so a missing spread would
    // silently drop whatever Core seeded onto the node.
    const s = stub()
    await buildStarterGraph(ZIMAGE, { x: 0, y: 0 })
    const genData = s.updateItem.mock.calls[0]?.[1].data as Record<string, unknown>
    const promptData = s.updateItem.mock.calls[1]?.[1].data as Record<string, unknown>
    expect(genData).toMatchObject({ keepMe: 1 })
    expect(genData.core).toEqual({ type: 'alibaba/z-image-turbo', params: ZIMAGE.params })
    expect(promptData).toMatchObject({ keepMe: 2, promptText: ZIMAGE.promptText })
  })

  it('does not add an undo entry for the param writes', async () => {
    // The adds already record; recording again would make Ctrl-Z walk through empty steps.
    const s = stub()
    await buildStarterGraph(ZIMAGE, { x: 0, y: 0 })
    for (const call of s.updateItem.mock.calls) expect(call[2]).toBe(false)
  })

  it('aborts without wiring when the model node cannot be created', async () => {
    const s = stub({ addCoreNode: vi.fn<Store['addCoreNode']>(async () => null) })
    expect(await buildStarterGraph(ZIMAGE, { x: 0, y: 0 })).toEqual([])
    expect(s.addPrompt).not.toHaveBeenCalled()
    expect(s.connect).not.toHaveBeenCalled()
    expect(useGenerationStore.getState().error).toMatch(/Inline Core/)
  })

  it('aborts without wiring when the prompt node cannot be created', async () => {
    const s = stub({ addPrompt: vi.fn<Store['addPrompt']>(async () => null) })
    expect(await buildStarterGraph(ZIMAGE, { x: 0, y: 0 })).toEqual([])
    expect(s.connect).not.toHaveBeenCalled()
  })

  it('builds nothing for a card that has no graph', async () => {
    const s = stub()
    expect(await buildStarterGraph(recipeFor('training')!, { x: 0, y: 0 })).toEqual([])
    expect(s.addCoreNode).not.toHaveBeenCalled()
  })
})
