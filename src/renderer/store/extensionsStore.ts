/**
 * Installed extensions, the published registry, and live install progress.
 *
 * Install is a two-step conversation: the first call returns `needsConsent` with a scan report and
 * installs nothing; the caller shows the report and calls again with the accepted rules.
 */
import { create } from 'zustand'
import type {
  ExtensionInfo,
  ExtensionTool,
  InstallProgressEvent,
  InstallSuccess,
  RegistryEntry,
  ScanReport,
  UpdateStatus,
} from '@shared/extensions'
import { studio } from '@/lib/studio'
import { useCoreNodesStore } from './coreNodesStore'

/** What the dialog is doing right now. */
export interface InstallState {
  source: string
  ref: string
  phase: InstallProgressEvent['phase'] | 'idle'
  fraction: number
  status: string
  /** Every phase seen so far, so the stepper can mark them done rather than only showing the
   * current one. Phases arrive in order and some (dependency install) are fast enough to miss. */
  seen: InstallProgressEvent['phase'][]
  error?: string
  /** Set when the install paused for consent, or was blocked outright. */
  report?: ScanReport
  conflicts?: { name: string; message: string }[]
  /** Set once the install succeeded, so the panel can confirm it instead of silently resetting. */
  done?: InstallSuccess
}

export type ExtensionsTab = 'installed' | 'available' | 'url'

interface ExtensionsState {
  open: boolean
  tab: ExtensionsTab
  extensions: ExtensionInfo[]
  tools: ExtensionTool[]
  canInstall: boolean
  registry: RegistryEntry[]
  registryStale: boolean
  loading: boolean
  /** Update status by extension id, filled in asynchronously after the dialog opens. */
  updates: Record<string, UpdateStatus>
  /** True once any operation reported that a restart is needed. */
  restartRequired: boolean
  install: InstallState | null

  openDialog: (tab?: ExtensionsTab) => void
  closeDialog: () => void
  setTab: (tab: ExtensionsTab) => void
  refresh: () => Promise<void>
  loadRegistry: (refresh?: boolean) => Promise<void>
  checkUpdates: () => Promise<void>
  beginInstall: (source: string, ref?: string, consents?: string[]) => Promise<void>
  clearInstall: () => void
  setEnabled: (extensionId: string, enabled: boolean) => Promise<void>
  setNodeEnabled: (extensionId: string, nodeType: string, enabled: boolean) => Promise<void>
  switchVersion: (extensionId: string, version: string) => Promise<void>
  uninstall: (extensionId: string) => Promise<void>
  onProgress: (e: InstallProgressEvent) => void
}

const IDLE: InstallState = { source: '', ref: '', phase: 'idle', fraction: 0, status: '', seen: [] }

export const useExtensionsStore = create<ExtensionsState>((set, get) => ({
  open: false,
  tab: 'installed',
  extensions: [],
  tools: [],
  canInstall: true,
  registry: [],
  registryStale: false,
  loading: false,
  updates: {},
  restartRequired: false,
  install: null,

  openDialog: (tab) => {
    set({ open: true, tab: tab ?? get().tab })
    void get().refresh()
    // Fired without awaiting: the list renders immediately and drift badges fill in later.
    void get().checkUpdates()
  },
  closeDialog: () => set({ open: false, install: null }),
  setTab: (tab) => {
    set({ tab })
    if (tab === 'available' && get().registry.length === 0) void get().loadRegistry()
  },

  refresh: async () => {
    set({ loading: true })
    const res = await studio().extensions.status()
    if (res.ok) {
      set({
        extensions: res.value.extensions,
        tools: res.value.tools,
        canInstall: res.value.canInstall,
        loading: false,
      })
    } else {
      set({ loading: false })
    }
  },

  checkUpdates: async () => {
    const res = await studio().extensions.checkUpdates()
    if (res.ok) {
      set({ updates: Object.fromEntries(res.value.map((u) => [u.extensionId, u])) })
    }
  },

  loadRegistry: async (refresh = false) => {
    const res = await studio().extensions.registryIndex(refresh)
    if (res.ok) set({ registry: res.value.entries, registryStale: res.value.stale })
  },

  beginInstall: async (source, ref = 'latest', consents) => {
    set({ install: { ...IDLE, source, ref, phase: 'fetch', status: 'Starting…', seen: [] } })
    const res = await studio().extensions.install(source, ref, consents)
    if (!res.ok) {
      set((s) => ({ install: { ...(s.install ?? IDLE), error: res.error, phase: 'idle' } }))
      return
    }
    const outcome = res.value
    if (!outcome.ok) {
      set((s) => ({
        install: {
          ...(s.install ?? IDLE),
          phase: outcome.phase,
          error: outcome.error,
          report: outcome.scan ?? undefined,
          conflicts: outcome.conflicts.map((c) => ({ name: c.name, message: c.message })),
        },
      }))
      return
    }
    if (outcome.needsConsent) {
      // Nothing was installed; the dialog shows the report and calls back with the rules.
      set((s) => ({
        install: { ...(s.install ?? IDLE), phase: 'scan', report: outcome.scan ?? undefined },
      }))
      return
    }
    set((s) => ({
      install: {
        ...(s.install ?? IDLE),
        phase: 'done',
        fraction: 1,
        status: 'Installed',
        seen: [...new Set([...(s.install?.seen ?? []), 'done' as const])],
        done: outcome,
      },
      restartRequired: s.restartRequired || outcome.restartRequired,
    }))
    await get().refresh()
    // New node types are live: refresh the palette so they appear in the add-node menu.
    await useCoreNodesStore.getState().load()
  },

  clearInstall: () => set({ install: null }),

  setEnabled: async (extensionId, enabled) => {
    const res = await studio().extensions.setEnabled(extensionId, enabled)
    if (res.ok && res.value.restartRequired) set({ restartRequired: true })
    await get().refresh()
    await useCoreNodesStore.getState().load()
  },

  setNodeEnabled: async (extensionId, nodeType, enabled) => {
    const res = await studio().extensions.setNodeEnabled(extensionId, nodeType, enabled)
    if (res.ok && res.value.restartRequired) set({ restartRequired: true })
    await get().refresh()
    await useCoreNodesStore.getState().load()
  },

  switchVersion: async (extensionId, version) => {
    const res = await studio().extensions.switchVersion(extensionId, version)
    if (res.ok) set({ restartRequired: true })
    await get().refresh()
  },

  uninstall: async (extensionId) => {
    const res = await studio().extensions.uninstall(extensionId)
    if (res.ok && res.value.restartRequired) set({ restartRequired: true })
    await get().refresh()
    await useCoreNodesStore.getState().load()
  },

  onProgress: (e) =>
    set((s) =>
      s.install
        ? {
            install: {
              ...s.install,
              phase: e.phase,
              fraction: e.fraction,
              status: e.status,
              seen: [...new Set([...s.install.seen, e.phase])],
            },
          }
        : {},
    ),
}))
