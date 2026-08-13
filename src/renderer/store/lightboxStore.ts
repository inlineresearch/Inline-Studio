/** A fullscreen media viewer ("lightbox") opened by double-clicking a node's image/video. */
import { create } from 'zustand'

export interface LightboxMedia {
  src: string
  kind: 'image' | 'video'
  name?: string
  /** Shown beside the first, for a Control LoRA pair: the two halves only mean anything together. */
  compare?: { src: string; kind: 'image' | 'video'; name?: string }
}

interface LightboxState {
  media: LightboxMedia | null
  open: (media: LightboxMedia) => void
  close: () => void
}

export const useLightboxStore = create<LightboxState>((set) => ({
  media: null,
  open: (media) => set({ media }),
  close: () => set({ media: null }),
}))
