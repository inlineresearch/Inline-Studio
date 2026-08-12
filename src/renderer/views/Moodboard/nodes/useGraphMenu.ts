/**
 * The graph actions behind the Run control's caret. Shared so the fal and Core nodes offer exactly
 * the same menu rather than drifting apart.
 */
import { useMemo } from 'react'
import type { RunMenuItem } from './NodeRunToolbar'
import { copyGraphJson, duplicateGraph, exportGraphJson, unsupportedTypes } from '../graphExport'

export function useGraphMenu(
  itemId: string,
  name: string,
): { items: RunMenuItem[]; note: string | undefined } {
  return useMemo(() => {
    const items: RunMenuItem[] = [
      { label: 'Copy graph JSON', onClick: () => void copyGraphJson(itemId) },
      { label: 'Export JSON…', onClick: () => exportGraphJson(itemId, name) },
      { label: 'Duplicate graph', onClick: () => void duplicateGraph(itemId) },
    ]
    // The importer rebuilds core/prompt/controlSpace/loader/frame only, so say so up front rather
    // than letting a re-import silently drop nodes.
    const unsupported = unsupportedTypes(itemId)
    const note =
      unsupported.length > 0
        ? `Re-importing skips: ${unsupported.join(', ')}. Duplicate keeps everything.`
        : undefined
    return { items, note }
  }, [itemId, name])
}
