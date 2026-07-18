/** Workspace-level UI state: which top-level mode/tab is active. */
import { create } from 'zustand'

export type WorkspaceMode = 'moodboard' | 'generate'

interface UiState {
  mode: WorkspaceMode
  /** Name of the most recently linked ComfyUI workflow, for the Generate banner. */
  linkedWorkflow: string | null
  /** The frame whose workflow is open in Generate - capture targets this frame. */
  activeFrameId: string | null
  /** The frame open in the right-side inspector drawer (null = closed). */
  inspectorFrameId: string | null
  /** Whether the Settings sidebar (fal API key, etc.) is open. */
  settingsOpen: boolean
  /** Currently selected moodboard node ids (mirrored from the canvas). */
  canvasSelection: string[]
  /** Flow-space center of the current canvas viewport (where the user is looking). */
  canvasCenter: { x: number; y: number }
  setMode: (mode: WorkspaceMode) => void
  setLinkedWorkflow: (name: string | null) => void
  setActiveFrame: (frameId: string | null) => void
  setInspectorFrame: (frameId: string | null) => void
  setSettingsOpen: (open: boolean) => void
  setCanvasSelection: (ids: string[]) => void
  setCanvasCenter: (c: { x: number; y: number }) => void
}

export const useUiStore = create<UiState>((set) => ({
  mode: 'moodboard',
  linkedWorkflow: null,
  activeFrameId: null,
  inspectorFrameId: null,
  settingsOpen: false,
  canvasSelection: [],
  canvasCenter: { x: 0, y: 0 },
  setMode: (mode) => set({ mode }),
  setLinkedWorkflow: (linkedWorkflow) => set({ linkedWorkflow }),
  setActiveFrame: (activeFrameId) => set({ activeFrameId }),
  setInspectorFrame: (inspectorFrameId) => set({ inspectorFrameId }),
  setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
  setCanvasSelection: (canvasSelection) => set({ canvasSelection }),
  setCanvasCenter: (canvasCenter) => set({ canvasCenter }),
}))
