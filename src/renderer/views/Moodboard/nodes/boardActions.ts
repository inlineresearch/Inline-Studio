/**
 * The board mutations the shared node chrome needs (move on chip-drag, resize, delete).
 *
 * Nodes render on two canvases now - the Studio moodboard and the Trainer tab's graph - backed by
 * different stores. `NodeFrame` / `NodeBadge` used to call `useMoodboardStore` directly, so a
 * Trainer node would have mutated Studio items. They go through this context instead: the Studio
 * canvas provides nothing and falls back to the moodboard store (unchanged behaviour), while the
 * Trainer canvas wraps its nodes in `BoardActionsContext.Provider` bound to its own store.
 */
import { createContext, useContext, useMemo } from 'react'
import type { MoodboardItemPatch } from '@shared/ipc'
import { useMoodboardStore } from '../../../store/moodboardStore'

export interface BoardActions {
  updateItem: (id: string, patch: MoodboardItemPatch) => Promise<void> | void
  deleteItem: (id: string) => Promise<void> | void
}

export const BoardActionsContext = createContext<BoardActions | null>(null)

export function useBoardActions(): BoardActions {
  const provided = useContext(BoardActionsContext)
  // Subscribed unconditionally to keep hook order stable; ignored when a provider is present.
  const updateItem = useMoodboardStore((s) => s.updateItem)
  const deleteItem = useMoodboardStore((s) => s.deleteItem)
  return useMemo(() => provided ?? { updateItem, deleteItem }, [provided, updateItem, deleteItem])
}
