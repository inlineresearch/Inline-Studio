/**
 * The Inline Core node palette: descriptors fetched from GET /v1/models (via the core IPC). The
 * canvas builds its add-node menu and renders nodes generically from these, so a new node type is a
 * Core change with no Storyline release. `registryVersion` lets us refetch when it changes.
 */
import { create } from 'zustand'
import type { NodeDescriptor } from '@shared/coreNodes'

interface CoreNodesState {
  descriptors: NodeDescriptor[]
  registryVersion: string | null
  running: boolean
  error: string | null
  load: () => Promise<void>
  byType: (type: string) => NodeDescriptor | undefined
}

export const useCoreNodesStore = create<CoreNodesState>((set, get) => ({
  descriptors: [],
  registryVersion: null,
  running: false,
  error: null,
  load: async () => {
    const status = await window.inlineStudio.core.status()
    const running = status.ok ? status.value.running : false
    const models = await window.inlineStudio.core.models()
    if (models.ok) {
      set({
        descriptors: models.value.models,
        registryVersion: models.value.registryVersion,
        running,
        error: null,
      })
    } else {
      set({ running, error: models.error })
    }
  },
  byType: (type) => get().descriptors.find((d) => d.type === type),
}))
