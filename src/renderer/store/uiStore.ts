/** Workspace-level UI state. */
import { create } from 'zustand'

interface UiState {
  /** The frame open in the right-side inspector drawer (null = closed). */
  inspectorFrameId: string | null
  /** Whether the Settings sidebar (fal API key, etc.) is open. */
  settingsOpen: boolean
  /** Currently selected moodboard node ids (mirrored from the canvas). */
  canvasSelection: string[]
  /** Flow-space center of the current canvas viewport (where the user is looking). */
  canvasCenter: { x: number; y: number }
  /** Somewhere the canvas should pan to. Cleared once it has. */
  reveal: { x: number; y: number } | null
  setInspectorFrame: (frameId: string | null) => void
  setSettingsOpen: (open: boolean) => void
  setCanvasSelection: (ids: string[]) => void
  setCanvasCenter: (c: { x: number; y: number }) => void
  /** Pan the canvas here. A graph dropped off-screen otherwise reads as nothing happening. */
  revealAt: (x: number, y: number) => void
  clearReveal: () => void
}

export const useUiStore = create<UiState>((set) => ({
  inspectorFrameId: null,
  settingsOpen: false,
  canvasSelection: [],
  canvasCenter: { x: 0, y: 0 },
  reveal: null,
  setInspectorFrame: (inspectorFrameId) => set({ inspectorFrameId }),
  setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
  setCanvasSelection: (canvasSelection) => set({ canvasSelection }),
  setCanvasCenter: (canvasCenter) => set({ canvasCenter }),
  revealAt: (x, y) => set({ reveal: { x, y } }),
  clearReveal: () => set({ reveal: null }),
}))
