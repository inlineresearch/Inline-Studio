/** Drag-and-drop helpers used within the renderer. */

/** Carries one or more asset ids when dragging from the Library. */
export const ASSET_DND_TYPE = 'application/x-inlinestudio-asset'

/** Encode the dragged asset ids onto a drag event's dataTransfer. */
export function setAssetDragPayload(dt: DataTransfer, assetIds: string[]): void {
  dt.setData(ASSET_DND_TYPE, JSON.stringify(assetIds))
  dt.effectAllowed = 'copy'
}

/** Decode dragged asset ids (tolerates a legacy single-id string payload). */
export function getAssetDragIds(dt: DataTransfer): string[] {
  const raw = dt.getData(ASSET_DND_TYPE)
  if (!raw) return []
  try {
    const parsed = JSON.parse(raw)
    if (Array.isArray(parsed)) return parsed.filter((x): x is string => typeof x === 'string')
  } catch {
    return [raw]
  }
  return []
}

/** Carries a frame id when dragging a frame from the Timeline tab onto the canvas. */
export const FRAME_DND_TYPE = 'application/x-inlinestudio-frame'

/** Encode the dragged frame id onto a drag event's dataTransfer. */
export function setFrameDragPayload(dt: DataTransfer, frameId: string): void {
  dt.setData(FRAME_DND_TYPE, frameId)
  dt.effectAllowed = 'copy'
}

/** Decode a dragged frame id, or null when the drag isn't a frame. */
export function getFrameDragId(dt: DataTransfer): string | null {
  return dt.getData(FRAME_DND_TYPE) || null
}

/**
 * Carries the producing frame id AND the specific take id when dragging a generated OUTPUT. Dropped
 * on the canvas it creates a NEW frame fed by that output; dropped on a node it feeds it as an input
 * (outputs also set the frame payload above). The take id lets the drop target pin the exact image
 * the user dragged (by making it the source frame's hero), rather than defaulting to the hero.
 */
export const OUTPUT_DND_TYPE = 'application/x-inlinestudio-output'

export function setOutputDragPayload(dt: DataTransfer, frameId: string, takeId?: string): void {
  dt.setData(OUTPUT_DND_TYPE, JSON.stringify({ frameId, takeId: takeId ?? null }))
}

function parseOutputPayload(dt: DataTransfer): { frameId: string; takeId: string | null } | null {
  const raw = dt.getData(OUTPUT_DND_TYPE)
  if (!raw) return null
  try {
    const p = JSON.parse(raw) as { frameId?: unknown; takeId?: unknown }
    if (typeof p.frameId !== 'string') return null
    return { frameId: p.frameId, takeId: typeof p.takeId === 'string' ? p.takeId : null }
  } catch {
    // Legacy payload: a bare frame id string.
    return { frameId: raw, takeId: null }
  }
}

/** Decode a dragged output's producing frame id, or null when the drag isn't an output. */
export function getOutputDragId(dt: DataTransfer): string | null {
  return parseOutputPayload(dt)?.frameId ?? null
}

/** Decode a dragged output's specific take id, or null (unknown / not an output drag). */
export function getOutputTakeId(dt: DataTransfer): string | null {
  return parseOutputPayload(dt)?.takeId ?? null
}
