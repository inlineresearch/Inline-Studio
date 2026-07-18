/**
 * Canvas display preferences (per-device, persisted to localStorage). These are pure view options
 * that don't belong in the project DB - e.g. the connector line style. Every consumer subscribes to
 * this store, so changing a preference re-renders live (all edges restyle at once).
 */
import { create } from 'zustand'

/** How connectors between nodes are drawn: `wave` = curved (bezier), `angled` = cornered (smooth step). */
export type EdgeStyle = 'wave' | 'angled'

const EDGE_STYLE_KEY = 'inline-studio.canvas.edgeStyle'

function loadEdgeStyle(): EdgeStyle {
  try {
    const v = localStorage.getItem(EDGE_STYLE_KEY)
    return v === 'angled' || v === 'wave' ? v : 'wave'
  } catch {
    return 'wave'
  }
}

interface CanvasPrefsState {
  edgeStyle: EdgeStyle
  setEdgeStyle: (style: EdgeStyle) => void
}

export const useCanvasPrefsStore = create<CanvasPrefsState>((set) => ({
  edgeStyle: loadEdgeStyle(),
  setEdgeStyle: (edgeStyle) => {
    try {
      localStorage.setItem(EDGE_STYLE_KEY, edgeStyle)
    } catch {
      // ignore a storage failure (private mode / quota) - the choice still applies this session.
    }
    set({ edgeStyle })
  },
}))
