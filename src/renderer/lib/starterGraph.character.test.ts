/** The character chain's handles must be the port ids the Core descriptors declare. */
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { buildCharacterStarter } from './starterGraph'
import { useMoodboardStore } from '../store/moodboardStore'

const added: { type: string; id: string }[] = []
const wires: string[] = []
const boardItems: { x: number; y: number; width: number; height: number }[] = []

beforeEach(() => {
  added.length = 0
  wires.length = 0
  boardItems.length = 0
  let n = 0
  const item = (type: string): { id: string; data: Record<string, unknown> } => {
    const id = `${type}-${++n}`
    added.push({ type, id })
    return { id, data: {} }
  }
  vi.spyOn(useMoodboardStore, 'getState').mockReturnValue({
    items: boardItems,
    addLoader: vi.fn(async () => item('loader')),
    addCoreNode: vi.fn(async (coreType: string) => item(coreType)),
    connect: vi.fn(async (from: string, to: string, source: string, target: string) => {
      wires.push(`${from.replace(/-\d+$/, '')}:${source} -> ${to.replace(/-\d+$/, '')}:${target}`)
    }),
  } as unknown as ReturnType<typeof useMoodboardStore.getState>)
})

describe('buildCharacterStarter', () => {
  it('drops the four nodes a reference character needs', async () => {
    const ids = await buildCharacterStarter({ x: 0, y: 0 })
    expect(added.map((a) => a.type)).toEqual([
      'loader',
      'character/encode',
      'character/references',
      'character/write',
    ])
    expect(ids).toHaveLength(4)
  })

  it('drops below whatever already occupies the viewport centre', async () => {
    // Dropping a chain on top of the user's existing graph reads as nodes vanishing.
    boardItems.push({ x: -200, y: -50, width: 400, height: 300 })
    const placed: number[] = []
    vi.mocked(useMoodboardStore.getState().addLoader).mockImplementation(async (_x, y) => {
      placed.push(y)
      return { id: 'loader-1', data: {} } as never
    })
    await buildCharacterStarter({ x: 0, y: 0 })
    expect(placed[0]).toBeGreaterThanOrEqual(250)
  })

  it('wires the identity to Write directly, not through the payload', async () => {
    // Payloads are compiled from the identity; Write needs both, so Encode fans out to each.
    await buildCharacterStarter({ x: 0, y: 0 })
    expect(wires).toEqual([
      'loader:image -> character/encode:images',
      'character/encode:character -> character/references:character',
      'character/encode:character -> character/write:character',
      'character/references:payload -> character/write:payloads',
    ])
  })
})
