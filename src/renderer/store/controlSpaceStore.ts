/**
 * Which Control Space node's full-screen 3D editor is open. Mirrors `lightboxStore` (a tiny toggle
 * store), so the editor is mounted once at app root and any Control Space node can open it by id.
 */
import { create } from 'zustand'

interface ControlSpaceState {
  /** The moodboard item id whose editor is open, or null when closed. */
  editingItemId: string | null
  open: (itemId: string) => void
  close: () => void
}

export const useControlSpaceStore = create<ControlSpaceState>((set) => ({
  editingItemId: null,
  open: (itemId) => set({ editingItemId: itemId }),
  close: () => set({ editingItemId: null }),
}))
