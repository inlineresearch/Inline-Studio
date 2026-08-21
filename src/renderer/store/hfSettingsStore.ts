/** Hugging Face token state; only whether one is configured ever crosses the wire. */
import { create } from 'zustand'
import { ipcErrorMessage } from '../lib/ipcError'
import { studio } from '@/lib/studio'

interface HfSettingsState {
  configured: boolean
  error: string | null
  load: () => Promise<void>
  setToken: (token: string) => Promise<boolean>
  clearToken: () => Promise<void>
}

export const useHfSettingsStore = create<HfSettingsState>((set) => ({
  configured: false,
  error: null,

  load: async () => {
    try {
      const res = await studio().hfSettings.status()
      if (res.ok) set({ configured: res.value.configured })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  setToken: async (token) => {
    try {
      const res = await studio().hfSettings.setToken(token)
      if (!res.ok) {
        set({ error: res.error })
        return false
      }
      set({ configured: res.value.configured, error: null })
      return true
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
      return false
    }
  },

  clearToken: async () => {
    try {
      const res = await studio().hfSettings.clearToken()
      if (!res.ok) return set({ error: res.error })
      set({ configured: res.value.configured, error: null })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },
}))
