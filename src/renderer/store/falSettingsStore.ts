/**
 * Renderer state for the fal.ai API key (bring-your-own-key). The key itself never comes back
 * across the bridge — only whether one is configured and if it's stored encrypted.
 */
import { create } from 'zustand'
import { ipcErrorMessage } from '../lib/ipcError'

interface FalSettingsState {
  configured: boolean
  encrypted: boolean
  error: string | null
  load: () => Promise<void>
  setApiKey: (key: string) => Promise<boolean>
  clearApiKey: () => Promise<void>
}

export const useFalSettingsStore = create<FalSettingsState>((set) => ({
  configured: false,
  encrypted: true,
  error: null,

  load: async () => {
    try {
      const res = await window.inlineStudio.falSettings.status()
      if (res.ok) set({ configured: res.value.configured, encrypted: res.value.encrypted })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  setApiKey: async (key) => {
    try {
      const res = await window.inlineStudio.falSettings.setApiKey(key)
      if (!res.ok) {
        set({ error: res.error })
        return false
      }
      set({ configured: res.value.configured, encrypted: res.value.encrypted, error: null })
      return true
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
      return false
    }
  },

  clearApiKey: async () => {
    try {
      const res = await window.inlineStudio.falSettings.clearApiKey()
      if (!res.ok) return set({ error: res.error })
      set({ configured: res.value.configured, encrypted: res.value.encrypted, error: null })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },
}))
