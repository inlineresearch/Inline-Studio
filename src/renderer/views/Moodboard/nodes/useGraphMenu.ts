/**
 * The graph actions behind the Run control's caret. Shared so the fal and Core nodes offer exactly
 * the same menu rather than drifting apart.
 */
import { useMemo } from 'react'
import type { RunMenuItem } from './NodeRunToolbar'
import { copyGraphJson, duplicateGraph, exportGraphJson, unsupportedTypes } from '../graphExport'
import { useCharacterStore } from '../../../store/characterStore'

export function useGraphMenu(
  itemId: string,
  name: string,
  /** The node's active image take, when it has one. Only an image can become a character. */
  takeId?: string,
): { items: RunMenuItem[]; note: string | undefined } {
  return useMemo(() => {
    const items: RunMenuItem[] = [
      { label: 'Copy graph JSON', onClick: () => void copyGraphJson(itemId) },
      { label: 'Export JSON…', onClick: () => exportGraphJson(itemId, name) },
      { label: 'Duplicate graph', onClick: () => void duplicateGraph(itemId) },
    ]
    if (takeId) {
      items.push({
        label: 'Save as character…',
        onClick: () => {
          const chosen = window.prompt('Name this character')?.trim()
          if (chosen) void useCharacterStore.getState().createFromTake(takeId, chosen)
        },
      })
    }
    // The importer rebuilds core/prompt/controlSpace/loader/frame only, so say so up front rather
    // than letting a re-import silently drop nodes.
    const unsupported = unsupportedTypes(itemId)
    const note =
      unsupported.length > 0
        ? `Re-importing skips: ${unsupported.join(', ')}. Duplicate keeps everything.`
        : undefined
    return { items, note }
  }, [itemId, name, takeId])
}
