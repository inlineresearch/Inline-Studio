/**
 * The board state and mutations a node needs, independent of which canvas it is on.
 *
 * Nodes render on the Studio moodboard and, until the surfaces merge, the Trainer graph - backed by
 * different stores. Reading a store directly meant a Trainer node would have mutated Studio items,
 * so everything goes through this context: the Studio canvas provides nothing and falls back to the
 * moodboard store, while the Trainer canvas binds its own.
 */
import { createContext, useContext, useMemo } from 'react'
import type { MoodboardConnector, MoodboardItem } from '@shared/types'
import type { MoodboardItemPatch } from '@shared/ipc'
import { useMoodboardStore } from '../../../store/moodboardStore'
import { useGenerationStore } from '../../../store/generationStore'

export interface BoardActions {
  items: MoodboardItem[]
  connectors: MoodboardConnector[]
  updateItem: (id: string, patch: MoodboardItemPatch) => Promise<void> | void
  /** Merge into an item's `data`; node selections (dataset, run, hyperparams) live there. */
  patchData: (id: string, data: Record<string, unknown>) => Promise<void> | void
  deleteItem: (id: string) => Promise<void> | void
  /** Node whose settings sidebar is open, since params live off the node face. */
  settingsItemId: string | null
  toggleSettings: (id: string) => void
}

export const BoardActionsContext = createContext<BoardActions | null>(null)

export function useBoardActions(): BoardActions {
  const provided = useContext(BoardActionsContext)
  // Subscribed unconditionally to keep hook order stable; ignored when a provider is present.
  const items = useMoodboardStore((s) => s.items)
  const connectors = useMoodboardStore((s) => s.connectors)
  const updateItem = useMoodboardStore((s) => s.updateItem)
  const patchItemData = useMoodboardStore((s) => s.patchItemData)
  const deleteItem = useMoodboardStore((s) => s.deleteItem)
  const settingsItemId = useGenerationStore((s) => s.settingsCoreItemId)
  const toggleSettings = useGenerationStore((s) => s.toggleCoreSettings)

  return useMemo(() => {
    if (provided) return provided
    return {
      items,
      connectors,
      updateItem,
      patchData: patchItemData,
      deleteItem,
      settingsItemId,
      toggleSettings,
    }
  }, [
    provided,
    items,
    connectors,
    updateItem,
    patchItemData,
    deleteItem,
    settingsItemId,
    toggleSettings,
  ])
}
