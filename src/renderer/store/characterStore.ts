/**
 * Saved characters: the `.char` library in `models/characters/`, which is global rather than
 * per-project. Work happens in Core via studio().characters; this only holds the list, the open
 * editor, and whether an encode is in flight (it runs two embedding passes, so it is not instant).
 */
import { create } from 'zustand'
import type { CharacterDetail, CharacterProgressEvent, CharacterSummary } from '@shared/types'
import type { CharacterBuildState } from './characterBuildState'

/** Which character the editor column is showing, and whether it is a new one. */
export type CharacterPanelMode =
  | { kind: 'create'; assetIds: string[] }
  | { kind: 'edit'; file: string }
import { ipcErrorMessage } from '../lib/ipcError'
import { studio } from '@/lib/studio'

interface CharacterState {
  characters: CharacterSummary[]
  /** The character open in the editor, with its reference URLs resolved. */
  editing: CharacterDetail | null
  loading: boolean
  /** An encode is running. Creating and editing both recompile, so both set this. */
  busy: boolean
  /** The running encode's latest phase, or null between encodes. */
  progress: CharacterProgressEvent | null
  /** Live training state per architecture, keyed by arch so each tab shows its own build. */
  builds: Record<string, CharacterBuildState>
  /** The editor column beside the side panel, or null when it is closed. */
  panel: CharacterPanelMode | null
  error: string | null

  load: () => Promise<void>
  create: (name: string, assetIds: string[], description: string) => Promise<boolean>
  createFromTake: (takeId: string, name: string) => Promise<boolean>
  open: (file: string) => Promise<void>
  closeEditor: () => void
  rename: (file: string, name: string) => Promise<void>
  setDescription: (file: string, description: string) => Promise<void>
  addRefs: (file: string, assetIds: string[]) => Promise<void>
  removeRef: (file: string, index: number) => Promise<void>
  remove: (file: string) => Promise<void>
  importFile: (file: File) => Promise<void>
  rebuild: (file: string) => Promise<void>
  cancelBuild: (arch: string) => Promise<void>
  setApplyMode: (file: string, arch: string, mode: 'reference' | 'lora') => Promise<void>
  openPanel: (mode: CharacterPanelMode) => void
  closePanel: () => void
  build: (
    file: string,
    arch: string,
    options: { steps: number; autoCaption: boolean },
  ) => Promise<boolean>
  reset: () => void
}

/** Merge an updated summary into the list without reordering it under the user's cursor. */
function merged(list: CharacterSummary[], updated: CharacterSummary): CharacterSummary[] {
  const index = list.findIndex((c) => c.file === updated.file)
  if (index < 0) return [updated, ...list]
  return list.map((c, i) => (i === index ? updated : c))
}

export const useCharacterStore = create<CharacterState>((set, get) => ({
  characters: [],
  editing: null,
  loading: false,
  busy: false,
  progress: null,
  builds: {},
  panel: null,
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

  create: async (name, assetIds, description) => {
    set({ busy: true, error: null, progress: null })
    try {
      const res = await studio().characters.create({ name, assetIds, description })
      if (!res.ok) {
        set({ busy: false, progress: null, error: res.error })
        return false
      }
      set((s) => ({ characters: merged(s.characters, res.value), busy: false, progress: null }))
      return true
    } catch (e) {
      set({ busy: false, progress: null, error: ipcErrorMessage(e) })
      return false
    }
  },

  createFromTake: async (takeId, name) => {
    set({ busy: true, error: null, progress: null })
    try {
      const res = await studio().characters.createFromTake(takeId, name)
      if (!res.ok) {
        set({ busy: false, progress: null, error: res.error })
        return false
      }
      set((s) => ({ characters: merged(s.characters, res.value), busy: false, progress: null }))
      return true
    } catch (e) {
      set({ busy: false, progress: null, error: ipcErrorMessage(e) })
      return false
    }
  },

  open: async (file) => {
    set({ error: null })
    try {
      const res = await studio().characters.get(file)
      if (!res.ok) return set({ error: res.error })
      set({ editing: res.value })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  closeEditor: () => set({ editing: null }),

  setApplyMode: async (file, arch, mode) => {
    await applyEdit(set, get, () => studio().characters.setApplyMode(file, arch, mode))
  },

  cancelBuild: async (arch) => {
    const runId = get().builds[arch]?.runId
    if (!runId) return
    await studio().training.cancel(runId)
  },

  openPanel: (mode) => {
    set({ panel: mode, error: null })
    if (mode.kind === 'edit') void get().open(mode.file)
  },

  closePanel: () => set({ panel: null, editing: null }),

  rename: async (file, name) => {
    await applyEdit(set, get, () => studio().characters.rename(file, name))
  },

  setDescription: async (file, description) => {
    await applyEdit(set, get, () => studio().characters.setDescription(file, description))
  },

  addRefs: async (file, assetIds) => {
    // No longer a recompile, so no busy state: the edit lands immediately.
    await applyEdit(set, get, () => studio().characters.addRefs(file, assetIds))
  },

  removeRef: async (file, index) => {
    await applyEdit(set, get, () => studio().characters.removeRef(file, index))
  },

  rebuild: async (file) => {
    await applyEdit(set, get, () => studio().characters.rebuild(file), true)
  },

  remove: async (file) => {
    set({ error: null })
    try {
      const res = await studio().characters.delete(file)
      if (!res.ok) return set({ error: res.error })
      set((s) => ({
        characters: s.characters.filter((c) => c.file !== file),
        editing: s.editing?.file === file ? null : s.editing,
      }))
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
      await get().load()
    } catch (e) {
      set({ busy: false, progress: null, error: ipcErrorMessage(e) })
    }
  },

  build: async (file, arch, options) => {
    set((s) => ({
      error: null,
      builds: {
        ...s.builds,
        [arch]: {
          phase: options.autoCaption ? 'captioning' : 'preparing',
          fraction: 0,
          step: 0,
          totalSteps: options.steps,
        },
      },
    }))
    try {
      const res = await studio().characters.build(file, arch, options)
      if (!res.ok) {
        set((s) => ({
          error: res.error,
          builds: { ...s.builds, [arch]: { ...s.builds[arch], phase: 'failed', error: res.error } },
        }))
        return false
      }
      // The run id is what ties every later training event back to this architecture's tab.
      set((s) => ({
        builds: {
          ...s.builds,
          [arch]: {
            ...s.builds[arch],
            phase: 'queued',
            runId: res.value.id,
            totalSteps: res.value.totalSteps,
          },
        },
      }))
      return true
    } catch (e) {
      const error = ipcErrorMessage(e)
      set((s) => ({
        error,
        builds: { ...s.builds, [arch]: { ...s.builds[arch], phase: 'failed', error } },
      }))
      return false
    }
  },

  reset: () =>
    set({
      characters: [],
      editing: null,
      error: null,
      busy: false,
      progress: null,
      builds: {},
      panel: null,
    }),
}))

type Setter = (
  partial: Partial<CharacterState> | ((s: CharacterState) => Partial<CharacterState>),
) => void

/** Every edit returns the rewritten summary, so the list and the open editor stay in step. */
async function applyEdit(
  set: Setter,
  get: () => CharacterState,
  call: () => Promise<{ ok: true; value: CharacterSummary } | { ok: false; error: string }>,
  recompiles = false,
): Promise<void> {
  set({ error: null, ...(recompiles ? { busy: true } : {}) })
  try {
    const res = await call()
    if (!res.ok) return set({ busy: false, progress: null, error: res.error })
    set((s) => ({ characters: merged(s.characters, res.value), busy: false, progress: null }))
    // Adding or removing a reference changes refUrls, so the editor has to be refetched.
    if (get().editing?.file === res.value.file) await get().open(res.value.file)
  } catch (e) {
    set({ busy: false, progress: null, error: ipcErrorMessage(e) })
  }
}

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
  // Training owns every phase after the queue, so the build tabs read its events rather than poll.
  const onTraining = studio().events.onTrainingProgress((e) => {
    patchBuild(e.runId, (b) => ({
      ...b,
      phase: 'training',
      fraction: e.fraction,
      step: e.step,
      totalSteps: e.totalSteps || b.totalSteps,
      status: e.status,
    }))
  })
  const onDone = studio().events.onTrainingDone((e) => {
    patchBuild(e.runId, (b) => ({ ...b, phase: 'done', fraction: 1 }))
    void useCharacterStore.getState().load()
  })
  const onError = studio().events.onTrainingError((e) => {
    patchBuild(e.runId, (b) => ({ ...b, phase: 'failed', error: e.error }))
  })
  return () => {
    onChanged()
    onProgress()
    onTraining()
    onDone()
    onError()
  }
}

/** Route a training event to whichever architecture started that run, ignoring everyone else's. */
function patchBuild(
  runId: string,
  update: (current: CharacterBuildState) => CharacterBuildState,
): void {
  const { builds } = useCharacterStore.getState()
  const arch = Object.keys(builds).find((key) => builds[key]?.runId === runId)
  if (!arch) return
  useCharacterStore.setState({ builds: { ...builds, [arch]: update(builds[arch]) } })
}
