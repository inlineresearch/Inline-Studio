/**
 * The ONLY bridge between renderer and main. Exposes a typed, minimal surface on
 * `window.inlineStudio` via contextBridge — no Node, no ipcRenderer, no raw channels
 * leak into the renderer (see CLAUDE.md security baseline + layering rule).
 */
import { contextBridge, ipcRenderer, webUtils } from 'electron'
import {
  IpcChannels,
  type InlineStudioApi,
  type CreateProjectInput,
  type CreateFolderInput,
  type MoodboardItemPatch,
} from '@shared/ipc'
import type {
  ComfyOutput,
  MoodboardItem,
  MoodboardConnector,
  UpdateAvailableEvent,
  UpdateProgressEvent,
  UpdateDownloadedEvent,
  TimelineProgressEvent,
  GenerationProgressEvent,
  GenerationNodeDoneEvent,
  GenerationDoneEvent,
  GenerationErrorEvent,
} from '@shared/types'
import type { IpcRendererEvent } from 'electron'

const api: InlineStudioApi = {
  project: {
    create: (input: CreateProjectInput) => ipcRenderer.invoke(IpcChannels.project.create, input),
    open: (path: string) => ipcRenderer.invoke(IpcChannels.project.open, path),
    openDialog: () => ipcRenderer.invoke(IpcChannels.project.openDialog),
    openZip: () => ipcRenderer.invoke(IpcChannels.project.openZip),
    listRecent: () => ipcRenderer.invoke(IpcChannels.project.listRecent),
    current: () => ipcRenderer.invoke(IpcChannels.project.current),
    mediaDirs: () => ipcRenderer.invoke(IpcChannels.project.mediaDirs),
    export: (path: string) => ipcRenderer.invoke(IpcChannels.project.export, path),
  },
  clipboard: {
    writeText: (text: string) => ipcRenderer.invoke(IpcChannels.clipboard.writeText, text),
  },
  assets: {
    importDialog: (folderId: string | null) =>
      ipcRenderer.invoke(IpcChannels.assets.importDialog, folderId),
    importPaths: (paths: string[], folderId: string | null) =>
      ipcRenderer.invoke(IpcChannels.assets.importPaths, paths, folderId),
    list: () => ipcRenderer.invoke(IpcChannels.assets.list),
    delete: (assetId: string) => ipcRenderer.invoke(IpcChannels.assets.delete, assetId),
  },
  folders: {
    list: () => ipcRenderer.invoke(IpcChannels.folders.list),
    create: (input: CreateFolderInput) => ipcRenderer.invoke(IpcChannels.folders.create, input),
    rename: (id: string, name: string) => ipcRenderer.invoke(IpcChannels.folders.rename, id, name),
    delete: (id: string) => ipcRenderer.invoke(IpcChannels.folders.delete, id),
  },
  frames: {
    list: () => ipcRenderer.invoke(IpcChannels.frames.list),
    importAsFrames: () => ipcRenderer.invoke(IpcChannels.frames.importAsFrames),
    addFromAsset: (assetId: string) => ipcRenderer.invoke(IpcChannels.frames.addFromAsset, assetId),
    rename: (id: string, name: string) => ipcRenderer.invoke(IpcChannels.frames.rename, id, name),
    reorder: (orderedIds: string[]) => ipcRenderer.invoke(IpcChannels.frames.reorder, orderedIds),
    delete: (id: string) => ipcRenderer.invoke(IpcChannels.frames.delete, id),
    clone: (id: string) => ipcRenderer.invoke(IpcChannels.frames.clone, id),
    unlink: (id: string) => ipcRenderer.invoke(IpcChannels.frames.unlink, id),
    setHero: (id: string, takeId: string | null) =>
      ipcRenderer.invoke(IpcChannels.frames.setHero, id, takeId),
    listTakes: (frameId: string) => ipcRenderer.invoke(IpcChannels.frames.listTakes, frameId),
    heroTakes: () => ipcRenderer.invoke(IpcChannels.frames.heroTakes),
    listInputs: () => ipcRenderer.invoke(IpcChannels.frames.listInputs),
    addInput: (frameId: string, assetId: string) =>
      ipcRenderer.invoke(IpcChannels.frames.addInput, frameId, assetId),
    addInputs: (frameId: string, assetIds: string[]) =>
      ipcRenderer.invoke(IpcChannels.frames.addInputs, frameId, assetIds),
    addSourceInput: (frameId: string, sourceFrameId: string) =>
      ipcRenderer.invoke(IpcChannels.frames.addSourceInput, frameId, sourceFrameId),
    removeInput: (frameId: string, assetId: string) =>
      ipcRenderer.invoke(IpcChannels.frames.removeInput, frameId, assetId),
    removeInputById: (frameId: string, inputId: string) =>
      ipcRenderer.invoke(IpcChannels.frames.removeInputById, frameId, inputId),
    reorderInputs: (frameId: string, orderedAssetIds: string[]) =>
      ipcRenderer.invoke(IpcChannels.frames.reorderInputs, frameId, orderedAssetIds),
    listAllTakes: () => ipcRenderer.invoke(IpcChannels.frames.listAllTakes),
    deleteTake: (takeId: string) => ipcRenderer.invoke(IpcChannels.frames.deleteTake, takeId),
    setFalParams: (frameId: string, params: Record<string, unknown>) =>
      ipcRenderer.invoke(IpcChannels.frames.setFalParams, frameId, params),
    setModel: (frameId: string, modelId: string) =>
      ipcRenderer.invoke(IpcChannels.frames.setModel, frameId, modelId),
    setProvider: (frameId: string, provider: 'comfy' | 'fal', modelId?: string) =>
      ipcRenderer.invoke(IpcChannels.frames.setProvider, frameId, provider, modelId),
  },
  generation: {
    run: (frameId: string) => ipcRenderer.invoke(IpcChannels.generation.run, frameId),
    runWorkflow: (itemId: string) => ipcRenderer.invoke(IpcChannels.generation.runWorkflow, itemId),
    cancel: (frameId?: string) => ipcRenderer.invoke(IpcChannels.generation.cancel, frameId),
    resumePending: () => ipcRenderer.invoke(IpcChannels.generation.resumePending),
  },
  falSettings: {
    status: () => ipcRenderer.invoke(IpcChannels.falSettings.status),
    setApiKey: (key: string) => ipcRenderer.invoke(IpcChannels.falSettings.setApiKey, key),
    clearApiKey: () => ipcRenderer.invoke(IpcChannels.falSettings.clearApiKey),
  },
  comfy: {
    status: () => ipcRenderer.invoke(IpcChannels.comfy.status),
    linkFrame: (frameId: string) => ipcRenderer.invoke(IpcChannels.comfy.linkFrame, frameId),
    uploadInputs: (frameId: string) => ipcRenderer.invoke(IpcChannels.comfy.uploadInputs, frameId),
    pullWorkflow: (frameId: string) => ipcRenderer.invoke(IpcChannels.comfy.pullWorkflow, frameId),
    saveLiveWorkflow: (frameId: string, workflow: unknown) =>
      ipcRenderer.invoke(IpcChannels.comfy.saveLiveWorkflow, frameId, workflow),
    pushWorkflow: (frameId: string) => ipcRenderer.invoke(IpcChannels.comfy.pushWorkflow, frameId),
    pullLatest: (frameId: string) => ipcRenderer.invoke(IpcChannels.comfy.pullLatest, frameId),
    latestRun: () => ipcRenderer.invoke(IpcChannels.comfy.latestRun),
    captureOutput: (frameId: string, output: ComfyOutput) =>
      ipcRenderer.invoke(IpcChannels.comfy.captureOutput, frameId, output),
  },
  settings: {
    get: () => ipcRenderer.invoke(IpcChannels.settings.get),
    setComfyUrl: (url: string) => ipcRenderer.invoke(IpcChannels.settings.setComfyUrl, url),
    setCoreUrl: (url: string) => ipcRenderer.invoke(IpcChannels.settings.setCoreUrl, url),
  },
  core: {
    status: () => ipcRenderer.invoke(IpcChannels.core.status),
    models: () => ipcRenderer.invoke(IpcChannels.core.models),
  },
  export: {
    exportFrames: () => ipcRenderer.invoke(IpcChannels.export.exportFrames),
  },
  moodboard: {
    list: () => ipcRenderer.invoke(IpcChannels.moodboard.list),
    addAsset: (assetId: string, x: number, y: number) =>
      ipcRenderer.invoke(IpcChannels.moodboard.addAsset, assetId, x, y),
    addText: (x: number, y: number) => ipcRenderer.invoke(IpcChannels.moodboard.addText, x, y),
    addFrameFromAsset: (assetId: string, x: number, y: number) =>
      ipcRenderer.invoke(IpcChannels.moodboard.addFrameFromAsset, assetId, x, y),
    addEmptyFrame: (x: number, y: number) =>
      ipcRenderer.invoke(IpcChannels.moodboard.addEmptyFrame, x, y),
    addFrameItem: (frameId: string, x: number, y: number) =>
      ipcRenderer.invoke(IpcChannels.moodboard.addFrameItem, frameId, x, y),
    addPreview: (x: number, y: number) =>
      ipcRenderer.invoke(IpcChannels.moodboard.addPreview, x, y),
    addLayer: (x: number, y: number) => ipcRenderer.invoke(IpcChannels.moodboard.addLayer, x, y),
    addDirector: (x: number, y: number) =>
      ipcRenderer.invoke(IpcChannels.moodboard.addDirector, x, y),
    addTrim: (x: number, y: number) => ipcRenderer.invoke(IpcChannels.moodboard.addTrim, x, y),
    addGenNode: (modelId: string, x: number, y: number) =>
      ipcRenderer.invoke(IpcChannels.moodboard.addGenNode, modelId, x, y),
    addPrompt: (x: number, y: number) => ipcRenderer.invoke(IpcChannels.moodboard.addPrompt, x, y),
    addCoreNode: (coreType: string, x: number, y: number) =>
      ipcRenderer.invoke(IpcChannels.moodboard.addCoreNode, coreType, x, y),
    updateItem: (id: string, patch: MoodboardItemPatch) =>
      ipcRenderer.invoke(IpcChannels.moodboard.updateItem, id, patch),
    deleteItem: (id: string) => ipcRenderer.invoke(IpcChannels.moodboard.deleteItem, id),
    importAndPlace: (x: number, y: number) =>
      ipcRenderer.invoke(IpcChannels.moodboard.importAndPlace, x, y),
    createConnector: (
      fromItemId: string,
      toItemId: string,
      sourceHandle: string | null = null,
      targetHandle: string | null = null,
    ) =>
      ipcRenderer.invoke(
        IpcChannels.moodboard.createConnector,
        fromItemId,
        toItemId,
        sourceHandle,
        targetHandle,
      ),
    deleteConnector: (id: string) => ipcRenderer.invoke(IpcChannels.moodboard.deleteConnector, id),
    setConnectorVolume: (id: string, volume: number) =>
      ipcRenderer.invoke(IpcChannels.moodboard.setConnectorVolume, id, volume),
    replaceBoard: (items: MoodboardItem[], connectors: MoodboardConnector[]) =>
      ipcRenderer.invoke(IpcChannels.moodboard.replaceBoard, items, connectors),
  },
  timeline: {
    resolve: (ownerItemId: string) => ipcRenderer.invoke(IpcChannels.timeline.resolve, ownerItemId),
    resolveTrim: (itemId: string) => ipcRenderer.invoke(IpcChannels.timeline.resolveTrim, itemId),
    setVolumes: (ownerItemId: string, l1Volume: number, l2Volume: number) =>
      ipcRenderer.invoke(IpcChannels.timeline.setVolumes, ownerItemId, l1Volume, l2Volume),
    buildPreview: (ownerItemId: string) =>
      ipcRenderer.invoke(IpcChannels.timeline.buildPreview, ownerItemId),
    export: (ownerItemId: string) => ipcRenderer.invoke(IpcChannels.timeline.export, ownerItemId),
    onProgress: (callback: (e: TimelineProgressEvent) => void) =>
      subscribe(IpcChannels.events.timelineProgress, callback),
  },
  dialog: {
    pickDirectory: () => ipcRenderer.invoke(IpcChannels.dialog.pickDirectory),
  },
  media: {
    save: (src: string, suggestedName: string) =>
      ipcRenderer.invoke(IpcChannels.media.save, src, suggestedName),
    copyImage: (src: string) => ipcRenderer.invoke(IpcChannels.media.copyImage, src),
  },
  shell: {
    openExternal: (url: string) => ipcRenderer.invoke(IpcChannels.shell.openExternal, url),
  },
  updates: {
    check: () => ipcRenderer.invoke(IpcChannels.updates.check),
    quitAndInstall: () => ipcRenderer.invoke(IpcChannels.updates.quitAndInstall),
  },
  app: {
    version: () => ipcRenderer.invoke(IpcChannels.app.version),
  },
  events: {
    onLibraryChanged: (callback: () => void) => {
      const listener = (): void => callback()
      ipcRenderer.on(IpcChannels.events.libraryChanged, listener)
      return () => ipcRenderer.removeListener(IpcChannels.events.libraryChanged, listener)
    },
    onGenerationProgress: (callback: (e: GenerationProgressEvent) => void) =>
      subscribe(IpcChannels.events.generationProgress, callback),
    onGenerationNodeDone: (callback: (e: GenerationNodeDoneEvent) => void) =>
      subscribe(IpcChannels.events.generationNodeDone, callback),
    onGenerationDone: (callback: (e: GenerationDoneEvent) => void) =>
      subscribe(IpcChannels.events.generationDone, callback),
    onGenerationError: (callback: (e: GenerationErrorEvent) => void) =>
      subscribe(IpcChannels.events.generationError, callback),
    onUpdateAvailable: (callback: (e: UpdateAvailableEvent) => void) =>
      subscribe(IpcChannels.events.updateAvailable, callback),
    onUpdateProgress: (callback: (e: UpdateProgressEvent) => void) =>
      subscribe(IpcChannels.events.updateProgress, callback),
    onUpdateDownloaded: (callback: (e: UpdateDownloadedEvent) => void) =>
      subscribe(IpcChannels.events.updateDownloaded, callback),
  },
  // Electron 32+: dropped File objects no longer expose `.path`; webUtils resolves it.
  getPathForFile: (file: File) => webUtils.getPathForFile(file),
}

/** Subscribe to a payload-carrying main→renderer event; returns an unsubscribe fn. */
function subscribe<T>(channel: string, callback: (payload: T) => void): () => void {
  const listener = (_e: IpcRendererEvent, payload: T): void => callback(payload)
  ipcRenderer.on(channel, listener)
  return () => ipcRenderer.removeListener(channel, listener)
}

contextBridge.exposeInMainWorld('inlineStudio', api)
