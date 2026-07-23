/** Workspace-level UI state. */
import { create } from 'zustand'

/** The two top-level surfaces: the node canvas (Studio) and the LoRA Trainer. */
export type WorkspaceTab = 'studio' | 'trainer'

interface UiState {
  /** Which top-level tab is showing. */
  activeTab: WorkspaceTab
  /** The frame open in the right-side inspector drawer (null = closed). */
  inspectorFrameId: string | null
  /** Whether the Settings sidebar (fal API key, etc.) is open. */
  settingsOpen: boolean
  /** Currently selected moodboard node ids (mirrored from the canvas). */
  canvasSelection: string[]
  /** Flow-space center of the current canvas viewport (where the user is looking). */
  canvasCenter: { x: number; y: number }
  setActiveTab: (tab: WorkspaceTab) => void
  setInspectorFrame: (frameId: string | null) => void
  setSettingsOpen: (open: boolean) => void
  setCanvasSelection: (ids: string[]) => void
  setCanvasCenter: (c: { x: number; y: number }) => void
}

export const useUiStore = create<UiState>((set) => ({
  activeTab: 'studio',
  inspectorFrameId: null,
  settingsOpen: false,
  canvasSelection: [],
  canvasCenter: { x: 0, y: 0 },
  setActiveTab: (activeTab) => set({ activeTab }),
  setInspectorFrame: (inspectorFrameId) => set({ inspectorFrameId }),
  setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
  setCanvasSelection: (canvasSelection) => set({ canvasSelection }),
  setCanvasCenter: (canvasCenter) => set({ canvasCenter }),
}))
