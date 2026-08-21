/**
 * Saved characters: the `.char` library in `models/characters/`, which is global rather than
 * per-project. This is the browser only - creating, editing and building all happen on the canvas,
 * through the character nodes, so nothing here writes to a character except the one-shot
 * "Save as character" a take offers.
 */
import { create } from 'zustand'
import type { CharacterProgressEvent, CharacterSummary } from '@shared/types'
import { ipcErrorMessage } from '../lib/ipcError'
import { studio } from '@/lib/studio'

interface CharacterState {
  characters: CharacterSummary[]
  loading: boolean
  /** An encode is running. It runs two embedding passes, so it is not instant. */
  busy: boolean
  /** The running encode's latest phase, or null between encodes. */
  progress: CharacterProgressEvent | null
  error: string | null

  load: () => Promise<void>
  createFromTake: (takeId: string, name: string) => Promise<boolean>
  remove: (file: string) => Promise<void>
  importFile: (file: File) => Promise<void>
  reset: () => void
}

//: What encoding needs on disk before it will run, so the popup can offer them by name.
const ENCODERS = [
  { filename: 'face_detection_yunet_2023mar.onnx', category: 'annotators' },
  { filename: 'face_recognition_sface_2021dec.onnx', category: 'annotators' },
  { filename: 'dinov2-base', category: 'annotators' },
]

/** True when the failure was a missing encoder and the model popup took over from it. */
async function offerEncoders(error: string): Promise<boolean> {
  if (!/character encoders/i.test(error)) return false
  const { checkModels } = await import('../lib/checkModels')
  await checkModels(ENCODERS, 'Creating a character needs its scoring encoders.')
  return true
}

/** Merge an updated summary into the list without reordering it under the user's cursor. */
function merged(list: CharacterSummary[], updated: CharacterSummary): CharacterSummary[] {
  const index = list.findIndex((c) => c.file === updated.file)
  if (index < 0) return [updated, ...list]
  return list.map((c, i) => (i === index ? updated : c))
}

export const useCharacterStore = create<CharacterState>((set) => ({
  characters: [],
  loading: false,
  busy: false,
  progress: null,
  error: null,

  load: async () => {
    set({ loading: true, error: null })
    try {
      const res = await studio().characters.list()
      if (!res.ok) return set({ loading: false, error: res.error })
      set({ characters: res.value, loading: false })
    } catch (e) {
      set({ loading: false, error: ipcErrorMessage(e) })
    }
  },

  createFromTake: async (takeId, name) => {
    set({ busy: true, error: null, progress: null })
    try {
      const res = await studio().characters.createFromTake(takeId, name)
      if (!res.ok) {
        const handled = await offerEncoders(res.error)
        set({ busy: false, progress: null, error: handled ? null : res.error })
        return false
      }
      set((s) => ({ characters: merged(s.characters, res.value), busy: false, progress: null }))
      return true
    } catch (e) {
      set({ busy: false, progress: null, error: ipcErrorMessage(e) })
      return false
    }
  },

  remove: async (file) => {
    set({ error: null })
    try {
      const res = await studio().characters.delete(file)
      if (!res.ok) return set({ error: res.error })
      set((s) => ({ characters: s.characters.filter((c) => c.file !== file) }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  importFile: async (file) => {
    set({ busy: true, error: null, progress: null })
    try {
      // /upload routes through the asset importer, which drops unknown extensions, so a .char
      // needs its own endpoint.
      const response = await fetch(`/upload/character?name=${encodeURIComponent(file.name)}`, {
        method: 'POST',
        body: file,
      })
      const result = (await response.json()) as { ok: boolean; error?: string }
      if (!result.ok)
        return set({ busy: false, progress: null, error: result.error ?? 'Import failed' })
      set({ busy: false, progress: null })
      await useCharacterStore.getState().load()
    } catch (e) {
      set({ busy: false, progress: null, error: ipcErrorMessage(e) })
    }
  },

  reset: () => set({ characters: [], error: null, busy: false, progress: null }),
}))

/**
 * Refresh when Core says the library moved. Called once from App's effect rather than at module
 * load, so the injected backend client is set first.
 */
export function subscribeCharacterChanges(): () => void {
  const onChanged = studio().events.onCharactersChanged(() => {
    void useCharacterStore.getState().load()
  })
  // Cleared on the last phase, since progress arrives while the call is still open.
  const onProgress = studio().events.onCharacterProgress((e) => {
    useCharacterStore.setState({ progress: e.fraction >= 1 ? null : e })
  })
  return () => {
    onChanged()
    onProgress()
  }
}
