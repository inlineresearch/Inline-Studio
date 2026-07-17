/** App settings (currently the ComfyUI backend URL). */
import { create } from 'zustand'
import { ipcErrorMessage } from '../lib/ipcError'
import { studio } from '@/lib/studio'

interface SettingsState {
  comfyUrl: string
  error: string | null
  load: () => Promise<void>
  setComfyUrl: (url: string) => Promise<void>
}

export const useSettingsStore = create<SettingsState>((set) => ({
  comfyUrl: '',
  error: null,

  load: async () => {
    try {
      const res = await studio().settings.get()
      if (res.ok) set({ comfyUrl: res.value.comfyUrl })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  setComfyUrl: async (url) => {
    try {
      const res = await studio().settings.setComfyUrl(url)
      if (!res.ok) return set({ error: res.error })
      set({ comfyUrl: res.value.comfyUrl, error: null })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },
}))
