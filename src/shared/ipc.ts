/**
 * The single typed contract for the renderer ↔ main IPC bridge.
 *
 * - `IpcChannels` are the only channel strings allowed (no stringly-typed
 *   `invoke('something')` scattered around — see CLAUDE.md).
 * - `InlineStudioApi` is the exact backend surface the web client implements against Core; the
 *   preload. The renderer imports this type; the main process implements it.
 */
import type {
  Project,
  RecentProject,
  Asset,
  AssetFolder,
  MoodboardItem,
  MoodboardConnector,
  MoodboardSnapshot,
  MoodboardItemData,
  DirectorTimeline,
  TrimResolved,
  TimelineProgressEvent,
  GenerationProgressEvent,
  GenerationNodeDoneEvent,
  GenerationDoneEvent,
  GenerationErrorEvent,
  Frame,
  Take,
  FrameInput,
  AppSettings,
  ApiKeyStatus,
  UpdateAvailableEvent,
  UpdateProgressEvent,
  UpdateDownloadedEvent,
  ComfyStatus,
  CoreStatus,
  ComfyOutput,
  ComfyRun,
  ExportResult,
  ProjectExportResult,
  ProjectMediaDirs,
} from './types'
import type { Result } from './result'
import type { CoreModels } from './coreNodes'

export const IpcChannels = {
  project: {
    create: 'project:create',
    open: 'project:open',
    openDialog: 'project:openDialog',
    openZip: 'project:openZip',
    listRecent: 'project:listRecent',
    current: 'project:current',
    mediaDirs: 'project:mediaDirs',
    export: 'project:export',
  },
  clipboard: {
    writeText: 'clipboard:writeText',
  },
  assets: {
    importDialog: 'assets:importDialog',
    importPaths: 'assets:importPaths',
    list: 'assets:list',
    delete: 'assets:delete',
  },
  folders: {
    list: 'folders:list',
    create: 'folders:create',
    rename: 'folders:rename',
    delete: 'folders:delete',
  },
  frames: {
    list: 'frames:list',
    importAsFrames: 'frames:importAsFrames',
    addFromAsset: 'frames:addFromAsset',
    rename: 'frames:rename',
    reorder: 'frames:reorder',
    delete: 'frames:delete',
    clone: 'frames:clone',
    unlink: 'frames:unlink',
    setHero: 'frames:setHero',
    listTakes: 'frames:listTakes',
    heroTakes: 'frames:heroTakes',
    listInputs: 'frames:listInputs',
    addInput: 'frames:addInput',
    addInputs: 'frames:addInputs',
    addSourceInput: 'frames:addSourceInput',
    removeInput: 'frames:removeInput',
    removeInputById: 'frames:removeInputById',
    reorderInputs: 'frames:reorderInputs',
    listAllTakes: 'frames:listAllTakes',
    deleteTake: 'frames:deleteTake',
    setFalParams: 'frames:setFalParams',
    setModel: 'frames:setModel',
    setProvider: 'frames:setProvider',
    resolveFalInputs: 'frames:resolveFalInputs',
  },
  generation: {
    /** Run a fal frame and its upstream chain (topologically). Streams progress via events. */
    run: 'generation:run',
    runWorkflow: 'generation:runWorkflow',
    /** Abort the in-flight generation run (optionally just one frame's). */
    cancel: 'generation:cancel',
    /** Re-poll + finish any runs that were in flight when the app last closed. */
    resumePending: 'generation:resumePending',
  },
  falSettings: {
    status: 'falSettings:status',
    setApiKey: 'falSettings:setApiKey',
    clearApiKey: 'falSettings:clearApiKey',
  },
  comfy: {
    status: 'comfy:status',
    linkFrame: 'comfy:linkFrame',
    uploadInputs: 'comfy:uploadInputs',
    pullWorkflow: 'comfy:pullWorkflow',
    saveLiveWorkflow: 'comfy:saveLiveWorkflow',
    pushWorkflow: 'comfy:pushWorkflow',
    pullLatest: 'comfy:pullLatest',
    latestRun: 'comfy:latestRun',
    captureOutput: 'comfy:captureOutput',
  },
  settings: {
    get: 'settings:get',
    setComfyUrl: 'settings:setComfyUrl',
    setCoreUrl: 'settings:setCoreUrl',
  },
  core: {
    status: 'core:status',
    models: 'core:models',
  },
  export: {
    exportFrames: 'export:exportFrames',
  },
  moodboard: {
    list: 'moodboard:list',
    addAsset: 'moodboard:addAsset',
    addText: 'moodboard:addText',
    addFrameFromAsset: 'moodboard:addFrameFromAsset',
    addEmptyFrame: 'moodboard:addEmptyFrame',
    addFrameItem: 'moodboard:addFrameItem',
    addPreview: 'moodboard:addPreview',
    addLayer: 'moodboard:addLayer',
    addDirector: 'moodboard:addDirector',
    addTrim: 'moodboard:addTrim',
    addGenNode: 'moodboard:addGenNode',
    addPrompt: 'moodboard:addPrompt',
    addCoreNode: 'moodboard:addCoreNode',
    updateItem: 'moodboard:updateItem',
    deleteItem: 'moodboard:deleteItem',
    importAndPlace: 'moodboard:importAndPlace',
    createConnector: 'moodboard:createConnector',
    deleteConnector: 'moodboard:deleteConnector',
    setConnectorVolume: 'moodboard:setConnectorVolume',
    replaceBoard: 'moodboard:replaceBoard',
  },
  timeline: {
    resolve: 'timeline:resolve',
    resolveTrim: 'timeline:resolveTrim',
    setVolumes: 'timeline:setVolumes',
    buildPreview: 'timeline:buildPreview',
    export: 'timeline:export',
  },
  dialog: {
    pickDirectory: 'dialog:pickDirectory',
  },
  media: {
    save: 'media:save',
    copyImage: 'media:copyImage',
  },
  shell: {
    openExternal: 'shell:openExternal',
  },
  updates: {
    /** Manually trigger a check for a newer published release. */
    check: 'updates:check',
    /** Quit and install a downloaded update (Windows/Linux). */
    quitAndInstall: 'updates:quitAndInstall',
  },
  app: {
    /** The running app version (from package.json). */
    version: 'app:version',
  },
  events: {
    /** Main → renderer: the asset library changed (e.g. a video poster/transcode is ready). */
    libraryChanged: 'events:libraryChanged',
    /** Main → renderer: director timeline render progress. */
    timelineProgress: 'events:timelineProgress',
    /** Main → renderer: fal generation lifecycle (per-node progress, node done, done, error). */
    generationProgress: 'events:generationProgress',
    generationNodeDone: 'events:generationNodeDone',
    generationDone: 'events:generationDone',
    generationError: 'events:generationError',
    /** Main → renderer: auto-update lifecycle. */
    updateAvailable: 'events:updateAvailable',
    updateProgress: 'events:updateProgress',
    updateDownloaded: 'events:updateDownloaded',
  },
} as const

/** Geometry/content fields a moodboard item update may change. */
export interface MoodboardItemPatch {
  x?: number
  y?: number
  width?: number
  height?: number
  rotation?: number
  zIndex?: number
  data?: MoodboardItemData
  /** Containing layer id, or null to detach from any layer. */
  parentId?: string | null
}

export interface CreateFolderInput {
  name: string
  /** Parent folder id, or null for a root-level folder. */
  parentId: string | null
}

/** A fal frame's inputs resolved for building its request: media as data URIs + the prompt text. */
export interface ResolvedFalInputs {
  images: string[]
  videos: string[]
  audios: string[]
  prompt: string | null
}

/** A prebuilt fal request the browser hands to the backend to run (fal defs are studio-side). */
export interface FalRunRequest {
  endpoint: string
  body: Record<string, unknown>
  outputKind: 'image' | 'video' | 'audio'
}

export interface CreateProjectInput {
  /** Display name; also used to derive the `.inlinestudio` folder name. */
  name: string
  /** Absolute parent directory the `.inlinestudio` folder is created in. */
  parentDir: string
}

/** The backend API surface — implemented by the web client (createWebClient) against Inline Core. */
export interface InlineStudioApi {
  project: {
    create(input: CreateProjectInput): Promise<Result<Project>>
    /** Open a `.inlinestudio` folder by absolute path. */
    open(path: string): Promise<Result<Project>>
    /** Show a native folder picker and open the chosen project. */
    openDialog(): Promise<Result<Project | null>>
    /** Show a .zip picker, extract the exported project beside the zip, and open it. */
    openZip(): Promise<Result<Project | null>>
    listRecent(): Promise<Result<RecentProject[]>>
    current(): Promise<Result<Project | null>>
    /** Absolute input/output dirs of the open project, for sharing with ComfyUI. */
    mediaDirs(): Promise<Result<ProjectMediaDirs>>
    /** Zip a project folder (by path) into a portable .zip; null if the save dialog is cancelled. */
    export(path: string): Promise<Result<ProjectExportResult | null>>
  }
  clipboard: {
    writeText(text: string): Promise<Result<void>>
  }
  assets: {
    /**
     * Show a native multi-file picker, copy chosen media into the project's library
     * under `folderId` (null = root), and return the new rows.
     */
    importDialog(folderId: string | null): Promise<Result<Asset[]>>
    /** Import media by absolute path (e.g. OS files dropped onto the canvas). */
    importPaths(paths: string[], folderId: string | null): Promise<Result<Asset[]>>
    /** All assets in the open project, newest first. */
    list(): Promise<Result<Asset[]>>
    /** Delete an asset (file + row); blocked if used by a frame. */
    delete(assetId: string): Promise<Result<void>>
  }
  folders: {
    /** All asset folders in the open project. */
    list(): Promise<Result<AssetFolder[]>>
    create(input: CreateFolderInput): Promise<Result<AssetFolder>>
    rename(id: string, name: string): Promise<Result<AssetFolder>>
    /** Delete a folder; its assets and subfolders move up to the parent. */
    delete(id: string): Promise<Result<void>>
  }
  frames: {
    /** All frames in the open project, in order. */
    list(): Promise<Result<Frame[]>>
    /** Import media via dialog and create a frame per file. */
    importAsFrames(): Promise<Result<Frame[]>>
    /** Create a frame from an existing library asset. */
    addFromAsset(assetId: string): Promise<Result<Frame>>
    rename(id: string, name: string): Promise<Result<Frame>>
    /** Persist a new left-to-right ordering. */
    reorder(orderedIds: string[]): Promise<Result<void>>
    delete(id: string): Promise<Result<void>>
    /** Duplicate a frame (its inputs + stored workflow); the clone starts unlinked. */
    clone(id: string): Promise<Result<Frame>>
    /** Detach the frame's ComfyUI workflow link. */
    unlink(id: string): Promise<Result<Frame>>
    /** Choose which take is the frame's Output (null clears it). */
    setHero(id: string, takeId: string | null): Promise<Result<Frame>>
    /** The frame's generated takes, newest first. */
    listTakes(frameId: string): Promise<Result<Take[]>>
    /** The hero (Output) take of every frame that has one. */
    heroTakes(): Promise<Result<Take[]>>
    /** All frame inputs across the project (group by frameId in the renderer). */
    listInputs(): Promise<Result<FrameInput[]>>
    /** Append a library asset as an input of the frame. */
    addInput(frameId: string, assetId: string): Promise<Result<FrameInput>>
    /** Append several library assets as inputs at once (atomic; skips duplicates). */
    addInputs(frameId: string, assetIds: string[]): Promise<Result<FrameInput[]>>
    /** Link another frame's output as an input (resolves to its hero take). */
    addSourceInput(frameId: string, sourceFrameId: string): Promise<Result<FrameInput>>
    /** Remove an input by its library asset id (Frame Inspector). */
    removeInput(frameId: string, assetId: string): Promise<Result<void>>
    /** Remove one input by its row id — works for asset AND flow-link inputs. */
    removeInputById(frameId: string, inputId: string): Promise<Result<void>>
    /** Persist a new input ordering for the frame. */
    reorderInputs(frameId: string, orderedAssetIds: string[]): Promise<Result<void>>
    /** All takes across the project (group by frameId in the renderer). */
    listAllTakes(): Promise<Result<Take[]>>
    /** Delete a generated take (clears it as hero if it was). */
    deleteTake(takeId: string): Promise<Result<void>>
    /** Persist a fal frame's param values (from the GenNode widgets). Returns the updated frame. */
    setFalParams(frameId: string, params: Record<string, unknown>): Promise<Result<Frame>>
    /** Switch a fal frame to a different model (resets params + output kind). Returns the frame. */
    setModel(frameId: string, modelId: string): Promise<Result<Frame>>
    /**
     * Resolve an `unset` chooser frame to an engine: `comfy` (embedded ComfyUI) or `fal` (a
     * declarative model — `modelId` defaults to the first registered model). Returns the frame.
     */
    setProvider(
      frameId: string,
      provider: 'comfy' | 'fal',
      modelId?: string,
    ): Promise<Result<Frame>>
    /** Resolve a fal frame's inputs (media as data URIs) + prompt, for building its request. */
    resolveFalInputs(frameId: string): Promise<Result<ResolvedFalInputs>>
  }
  generation: {
    /**
     * Run a fal frame. On the single-process (web) backend the browser passes a prebuilt `request`
     * (fal defs are studio-side); the Electron backend builds it server-side and ignores `request`.
     * Resolves immediately; progress arrives via events.
     */
    run(frameId: string, request?: FalRunRequest): Promise<Result<void>>
    runWorkflow(itemId: string): Promise<Result<void>>
    /** Abort the in-flight run — a specific frame's, or all when no id is given. */
    cancel(frameId?: string): Promise<Result<void>>
    /** Re-poll + finish any generations that were in flight when the app last closed. */
    resumePending(): Promise<Result<void>>
  }
  falSettings: {
    /** Is a fal API key saved, and is it stored encrypted? */
    status(): Promise<Result<ApiKeyStatus>>
    /** Store the fal API key (encrypted via safeStorage when available). */
    setApiKey(key: string): Promise<Result<ApiKeyStatus>>
    /** Forget the stored fal key. */
    clearApiKey(): Promise<Result<ApiKeyStatus>>
  }
  comfy: {
    /** Is the configured ComfyUI reachable? */
    status(): Promise<Result<ComfyStatus>>
    /** Create/ensure this frame's linked ComfyUI workflow; returns the updated frame. */
    linkFrame(frameId: string): Promise<Result<Frame>>
    /** Upload the frame's input assets to ComfyUI (cloud-safe); returns stored names. */
    uploadInputs(frameId: string): Promise<Result<string[]>>
    /** Pull the frame's workflow from ComfyUI into the project copy; true if changed. */
    pullWorkflow(frameId: string): Promise<Result<boolean>>
    /**
     * Capture the live (possibly unsaved) graph serialized off the ComfyUI canvas into
     * the project copy. Returns the updated frame if anything changed, else null.
     */
    saveLiveWorkflow(frameId: string, workflow: unknown): Promise<Result<Frame | null>>
    /** Push the project's copy of the frame's workflow to ComfyUI. */
    pushWorkflow(frameId: string): Promise<Result<void>>
    /** Pull ComfyUI's latest output and attach it to the frame as its Output take. */
    pullLatest(frameId: string): Promise<Result<Take>>
    /** The most recent ComfyUI run + all its output files (for the capture strip). */
    latestRun(): Promise<Result<ComfyRun | null>>
    /** Download a specific ComfyUI output and attach it to the frame as a take. */
    captureOutput(frameId: string, output: ComfyOutput): Promise<Result<Take>>
  }
  settings: {
    get(): Promise<Result<AppSettings>>
    setComfyUrl(url: string): Promise<Result<AppSettings>>
    setCoreUrl(url: string): Promise<Result<AppSettings>>
  }
  core: {
    status(): Promise<Result<CoreStatus>>
    models(): Promise<Result<CoreModels>>
  }
  export: {
    /** Pick a folder and write each frame's Output in order; null if cancelled. */
    exportFrames(): Promise<Result<ExportResult | null>>
  }
  moodboard: {
    /** The full board (items + connectors) for the open project. */
    list(): Promise<Result<MoodboardSnapshot>>
    /** Place an existing library asset on the board at (x, y). */
    addAsset(assetId: string, x: number, y: number): Promise<Result<MoodboardItem>>
    /** Add a new editable text item at (x, y). */
    addText(x: number, y: number): Promise<Result<MoodboardItem>>
    /** Create a frame from a library asset AND place a frame node on the canvas. */
    addFrameFromAsset(assetId: string, x: number, y: number): Promise<Result<MoodboardItem>>
    /** Create an empty frame AND place a frame node on the canvas at (x, y). */
    addEmptyFrame(x: number, y: number): Promise<Result<MoodboardItem>>
    /** Place an existing frame as a node on the canvas. */
    addFrameItem(frameId: string, x: number, y: number): Promise<Result<MoodboardItem>>
    /** Add an empty Preview node at (x, y). */
    addPreview(x: number, y: number): Promise<Result<MoodboardItem>>
    /** Add a resizable layer group container at (x, y). */
    addLayer(x: number, y: number): Promise<Result<MoodboardItem>>
    /** Add a video-director node (timeline-in-a-node) at (x, y). */
    addDirector(x: number, y: number): Promise<Result<MoodboardItem>>
    /** Add an "Edit Video/Audio" (trim) node at (x, y). */
    addTrim(x: number, y: number): Promise<Result<MoodboardItem>>
    /** Create a fal generation frame for `modelId` AND place its node on the canvas at (x, y). */
    addGenNode(modelId: string, x: number, y: number): Promise<Result<MoodboardItem>>
    /** Add a text-prompt node (feeds a Generate node's prompt input) at (x, y). */
    addPrompt(x: number, y: number): Promise<Result<MoodboardItem>>
    addCoreNode(coreType: string, x: number, y: number): Promise<Result<MoodboardItem>>
    updateItem(id: string, patch: MoodboardItemPatch): Promise<Result<MoodboardItem>>
    deleteItem(id: string): Promise<Result<void>>
    /** Import media into the shared library AND place it on the board near (x, y). */
    importAndPlace(x: number, y: number): Promise<Result<MoodboardItem[]>>
    createConnector(
      fromItemId: string,
      toItemId: string,
      sourceHandle?: string | null,
      targetHandle?: string | null,
    ): Promise<Result<MoodboardConnector>>
    deleteConnector(id: string): Promise<Result<void>>
    /** Set a connector's per-input audio volume (0..1) — director L1 inputs. */
    setConnectorVolume(id: string, volume: number): Promise<Result<void>>
    /** Replace the entire board (used by canvas undo/redo). */
    replaceBoard(items: MoodboardItem[], connectors: MoodboardConnector[]): Promise<Result<void>>
  }
  timeline: {
    /** The derived timeline (video + L2 audio + volumes) for a director node. */
    resolve(ownerItemId: string): Promise<Result<DirectorTimeline>>
    /** The full source media behind a trim node's input (for its UI); null if unwired. */
    resolveTrim(itemId: string): Promise<Result<TrimResolved | null>>
    /** Set the L1/L2 layer volumes (0..1) on a director node. */
    setVolumes(ownerItemId: string, l1Volume: number, l2Volume: number): Promise<Result<void>>
    /** Render a low-res proxy preview; returns its project-relative path (null if empty). */
    buildPreview(ownerItemId: string): Promise<Result<string | null>>
    /** Export the timeline to a user-chosen MP4; returns the path written (null if cancelled). */
    export(ownerItemId: string): Promise<Result<string | null>>
    /** Subscribe to render progress (0..1). Returns an unsubscribe fn. */
    onProgress(cb: (e: TimelineProgressEvent) => void): () => void
  }
  dialog: {
    /** Native folder picker; returns the chosen absolute path or null if cancelled. */
    pickDirectory(): Promise<Result<string | null>>
  }
  media: {
    /**
     * Save a project media file (an image/video/audio take or asset) to a location
     * the user picks. `src` is its media URL or project-relative path; `suggestedName`
     * seeds the save dialog's filename (the source extension is appended if missing).
     * Resolves to `true` if saved, `false` if the dialog was cancelled.
     */
    save(src: string, suggestedName: string): Promise<Result<boolean>>
    /** Copy an image (by media URL or project-relative path) to the system clipboard. */
    copyImage(src: string): Promise<Result<void>>
  }
  shell: {
    /** Open an http(s) URL in the user's default browser. */
    openExternal(url: string): Promise<Result<void>>
  }
  updates: {
    /** Trigger a check for a newer published release. */
    check(): Promise<Result<void>>
    /** Quit and install a downloaded update (Windows/Linux). */
    quitAndInstall(): Promise<Result<void>>
  }
  app: {
    /** The running app version (from package.json). */
    version(): Promise<Result<string>>
  }
  /** Resolve the absolute path of a File dropped from the OS (Electron webUtils). Sync. */
  getPathForFile(file: File): string
  events: {
    /** Subscribe to "asset library changed" pushes from main. Returns an unsubscribe fn. */
    onLibraryChanged(callback: () => void): () => void
    /** Subscribe to fal generation lifecycle pushes. Each returns an unsubscribe fn. */
    onGenerationProgress(callback: (e: GenerationProgressEvent) => void): () => void
    onGenerationNodeDone(callback: (e: GenerationNodeDoneEvent) => void): () => void
    onGenerationDone(callback: (e: GenerationDoneEvent) => void): () => void
    onGenerationError(callback: (e: GenerationErrorEvent) => void): () => void
    /** Subscribe to auto-update lifecycle pushes. Each returns an unsubscribe fn. */
    onUpdateAvailable(callback: (e: UpdateAvailableEvent) => void): () => void
    onUpdateProgress(callback: (e: UpdateProgressEvent) => void): () => void
    onUpdateDownloaded(callback: (e: UpdateDownloadedEvent) => void): () => void
  }
}
