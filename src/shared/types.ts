/**
 * Domain types shared by the main and renderer processes.
 *
 * The mental model (see CLAUDE.md):
 *   Project → Sequence → Frame → Take[]
 *   A Frame is a *slot with a history of takes*, never a single file.
 *   The timeline points at a Frame's `heroTakeId` (the current chosen take).
 */

export type FrameKind = 'image' | 'video' | 'audio'
export type AssetKind = 'image' | 'video' | 'audio'

/** A project is a portable `.inlinestudio` folder; this is its DB-backed metadata. */
export interface Project {
  id: string
  name: string
  /** Absolute path to the `.inlinestudio` folder on disk. */
  path: string
  createdAt: number
  updatedAt: number
}

/** Lightweight entry for the "recent projects" list (stored in app userData). */
export interface RecentProject {
  name: string
  path: string
  lastOpenedAt: number
}

export interface Sequence {
  id: string
  projectId: string
  name: string
  /** Order within the project. */
  position: number
}

export interface Frame {
  id: string
  sequenceId: string
  name: string
  kind: FrameKind
  /** Order within the sequence. */
  position: number
  /** The imported source asset this frame edits (its Input row), if any. */
  inputAssetId: string | null
  /** Currently chosen take placed on the timeline, if any. */
  heroTakeId: string | null
  /**
   * Which generation engine backs this frame. `unset` = a fresh chooser node that hasn't picked an
   * engine yet (renders the Link-ComfyUI / Generate-with-Fal chooser). `comfy` = the embedded
   * ComfyUI workflow (every legacy frame). `fal` = a declarative fal.ai node (see `modelId`/`params`).
   * `core` = an Inline Core node: same `modelId`/`params`, run by the Inline Core engine.
   */
  provider: 'comfy' | 'fal' | 'core' | 'unset'
  /** For `provider:'fal'`: the registry model id (e.g. `openai/gpt-image-2`). Null for comfy. */
  modelId: string | null
  /** For `provider:'fal'`: the node's editable param values (keyed by the NodeDef param keys). */
  params: Record<string, unknown>
  /** Workflow template this frame generates with, if chosen. */
  workflowTemplateId: string | null
  /** The ComfyUI workflow (userdata name) this frame is linked to, if any. */
  comfyWorkflowName: string | null
  /**
   * True once a real (non-seed) workflow graph has been captured for this frame — i.e.
   * the user has actually built something, not just clicked Link (which seeds a Note).
   * Lets the UI distinguish "linked but empty" from "ready to generate".
   */
  comfyWorkflowReady: boolean
  createdAt: number
  updatedAt: number
}

/**
 * One of a frame's inputs (a frame can have several). An input is either a library
 * asset (`assetId`) or a live link to another frame's output (`sourceFrameId`) — the
 * latter resolves to that frame's hero take at generate time (the refine/flow link).
 */
export interface FrameInput {
  id: string
  frameId: string
  assetId: string | null
  sourceFrameId: string | null
  position: number
}

/** Every ComfyUI render of a frame becomes an immutable Take. */
export interface Take {
  id: string
  frameId: string
  /** Relative path under the project's `takes/` folder. */
  filePath: string
  kind: AssetKind
  /** Snapshot of the params used to generate this take (JSON). */
  params: Record<string, unknown>
  /** ComfyUI prompt id that produced this take, if generated. */
  comfyPromptId: string | null
  createdAt: number
}

/** A logical folder in the asset library (a tree; physical files stay flat on disk). */
export interface AssetFolder {
  id: string
  projectId: string
  name: string
  /** Parent folder id, or null for a root-level folder. */
  parentId: string | null
  createdAt: number
}

/** Imported media in the project library. */
export interface Asset {
  id: string
  projectId: string
  /** Logical folder this asset lives in, or null for the library root. */
  folderId: string | null
  name: string
  /** Relative path under the project's `assets/` folder. */
  filePath: string
  kind: AssetKind
  /** Poster image (first frame) for videos, so they always render. Relative path. */
  thumbPath: string | null
  /** A Chromium-playable transcode for videos in codecs the UI can't decode. Relative. */
  previewPath: string | null
  createdAt: number
}

/**
 * An item on the unified canvas. Beyond ideation items (asset/text), the canvas
 * also hosts the production graph: `frame` nodes (input/output handles), `layer`
 * group containers, and `preview` nodes that display a connected frame's output.
 */
export type MoodboardItemType =
  | 'asset'
  | 'text'
  | 'frame'
  | 'layer'
  | 'preview'
  | 'director'
  | 'trim'
  /** A text-prompt node whose output feeds a Generate node's prompt input. */
  | 'prompt'
  /** A low-level Inline Core graph node (load/sample/encode/vae); ephemeral, not a Frame. */
  | 'core'

/** Output settings for a video-director node (stored in its moodboard item data). */
export interface DirectorItemData {
  /** Composition width in px. */
  width: number
  /** Composition height in px. */
  height: number
  /** Frames per second. */
  fps: number
}

/**
 * One derived clip on a director node's timeline. The video layer and L1 audio are
 * derived from the videos wired into the node; L2 is the user's wired audio. Positions
 * are sequential (seconds); nothing here is persisted — it's recomputed from the wired
 * connections each time.
 */
export interface DirectorClip {
  /** Stable key for React (the source frame/asset id). */
  key: string
  /** The connector feeding this input (for per-input volume edits). */
  connectorId: string
  /** Per-input audio volume 0..1 (L1 = the video's extracted audio). */
  volume: number
  /** Source frame id (for the "Frame X" tag + canvas navigation); null for asset clips. */
  frameId: string | null
  /** Display label (frame name / asset name). */
  label: string
  kind: AssetKind
  /** Position on the layer, in seconds. */
  startTime: number
  /** Clip length, in seconds. */
  duration: number
  /** Project-relative waveform peaks JSON for this clip's audio, if any. */
  audioPeaks: string | null
  /** Fraction [start, end] of the source peaks this clip spans (a trim window; full = 0..1). */
  peaksStart: number
  peaksEnd: number
  /** Project-relative thumbnail: a filmstrip PNG for video, the still image for image clips. */
  thumbnail: string | null
}

/** The resolved source media behind a trim ("Edit Video/Audio") node, for its UI. */
export interface TrimResolved {
  /** Source frame/asset id. */
  key: string
  kind: AssetKind
  label: string
  /** Full source duration in seconds (0 if unknown). */
  durationSec: number
  /** Project-relative media path (for an in-node preview element). */
  mediaPath: string
  /** Project-relative filmstrip PNG (video), or null. */
  thumbnail: string | null
  /** Project-relative waveform peaks JSON (audio), or null. */
  audioPeaks: string | null
}

/** The derived, display-ready timeline for a director node (recomputed from connections). */
export interface DirectorTimeline {
  /** Video layer clips (in slot order). */
  video: DirectorClip[]
  /** Audio layer 2 clips (the user's wired audio). */
  l2: DirectorClip[]
  /** Volume of the extracted video audio (L1), 0..1. */
  l1Volume: number
  /** Volume of the user audio (L2), 0..1. */
  l2Volume: number
}

export interface TextItemData {
  text: string
  fontSize: number
  bold: boolean
  italic: boolean
  underline: boolean
  color: string
  align: 'left' | 'center' | 'right'
  /** Optional http(s) URL — renders the text as a clickable link. */
  link?: string
}

/** Type-specific payload for a moodboard item (text styling, layer/director settings). */
export interface MoodboardItemData {
  text?: TextItemData
  /** Display name for a layer group or director node. */
  name?: string
  /** Accent color (hex) for a layer group. */
  color?: string
  /** Output settings for a director node. */
  director?: DirectorItemData
  /** Project-relative path of a director node's last-built proxy preview MP4. */
  directorPreview?: string
  /** Project-relative path of a director node's last full-res Export MP4 (shown in a wired preview). */
  directorExport?: string
  /** Director node: volume of the extracted video audio layer (L1), 0..1 (default 1). */
  l1Volume?: number
  /** Director node: volume of the user audio layer (L2), 0..1 (default 1). */
  l2Volume?: number
  /** Trim ("Edit Video/Audio") node: in/out window in seconds (outPoint <= inPoint = "to end"). */
  trim?: { inPoint: number; outPoint: number }
  /** Prompt node: the text it feeds into a connected Generate node's prompt input. */
  promptText?: string
  /** Core graph node: the Inline Core node type + its param values (see coreNodes.ts). */
  core?: {
    type: string
    params: Record<string, unknown>
    /** The latest media this node produced (media-output nodes only). Project-relative path. */
    output?: { takeId: string; filePath: string; kind: 'image' | 'video' | 'audio' }
  }
}

export interface MoodboardItem {
  id: string
  projectId: string
  type: MoodboardItemType
  /** Set when type === 'asset'. */
  assetId: string | null
  /** Set when type === 'frame' (and reused by 'preview' resolution). */
  frameId: string | null
  /** Containing layer item id, if this item lives inside a layer group. */
  parentId: string | null
  data: MoodboardItemData
  x: number
  y: number
  width: number
  height: number
  rotation: number
  zIndex: number
  createdAt: number
  updatedAt: number
}

/** An arrow/line connecting two moodboard items. */
export interface MoodboardConnector {
  id: string
  projectId: string
  fromItemId: string
  toItemId: string
  label: string | null
  data: Record<string, unknown>
  createdAt: number
}

/** The full board state for a project. */
export interface MoodboardSnapshot {
  items: MoodboardItem[]
  connectors: MoodboardConnector[]
}

/** Main → renderer: director timeline render progress (0..1), correlated by node id. */
export interface TimelineProgressEvent {
  ownerItemId: string
  fraction: number
}

/** Main → renderer: per-node generation progress (0..1), correlated by the running frame's id. */
export interface GenerationProgressEvent {
  frameId: string
  fraction: number
  /** Optional short status label from the fal queue (e.g. "in queue", "generating"). */
  status?: string
}

/** Main → renderer: a node in the run finished and produced a take. */
export interface GenerationNodeDoneEvent {
  frameId: string
  takeId: string
}

/** Main → renderer: the whole run (target frame + its upstream chain) completed. */
export interface GenerationDoneEvent {
  targetFrameId: string
}

/** Main → renderer: the run failed; `frameId` is the node that failed, if any. */
export interface GenerationErrorEvent {
  targetFrameId: string
  frameId?: string
  error: string
}

/** Main → renderer: progress (0..1) of an explicit model download (the node's model popup). */
export interface ModelDownloadProgressEvent {
  nodeType: string
  componentId: string
  fraction: number
  status?: string
}

/** Main → renderer: a model component finished downloading (files are now under models/). */
export interface ModelDownloadDoneEvent {
  nodeType: string
  componentId: string
}

/** Main → renderer: a model download failed. */
export interface ModelDownloadErrorEvent {
  nodeType: string
  componentId: string
  error: string
}

export interface WorkflowTemplate {
  id: string
  projectId: string
  name: string
  /** Raw ComfyUI workflow JSON. */
  graph: Record<string, unknown>
  /** Param schema marking which node inputs are user-editable. */
  params: WorkflowParam[]
}

export interface WorkflowParam {
  key: string
  label: string
  type: 'text' | 'number' | 'seed' | 'image' | 'model'
  /** Path into the graph JSON this param drives, e.g. "6.inputs.text". */
  nodePath: string
  default?: string | number
}

/** App-global settings (stored in Electron userData, not per-project). */
export interface AppSettings {
  /** The ComfyUI backend Inline Studio talks to and embeds. */
  comfyUrl: string
  /** The Inline Core (/v1) engine URL. */
  coreUrl: string
}

/** Whether an API key (e.g. fal.ai) is saved, and how it's stored. */
export interface ApiKeyStatus {
  /** A key is saved. */
  configured: boolean
  /** False when the OS lacks a secure keystore and the key is stored plaintext. */
  encrypted: boolean
}

/** Auto-update events (main → renderer). */
export interface UpdateAvailableEvent {
  version: string
  /** macOS (unsigned) can't self-install, so the renderer opens the releases page instead. */
  notifyOnly: boolean
}
export interface UpdateProgressEvent {
  /** 0–100. */
  percent: number
  transferred: number
  total: number
}
export interface UpdateDownloadedEvent {
  version: string
}

/** Result of pinging the configured ComfyUI backend. */
export interface ComfyStatus {
  running: boolean
  url: string
}

/** Result of pinging the configured Inline Core engine. */
export interface CoreStatus {
  running: boolean
  url: string
}

/** A single output file produced by a ComfyUI run. */
export interface ComfyOutput {
  filename: string
  subfolder: string
  type: string
  kind: AssetKind
  /** A `${comfyUrl}/view?...` URL for displaying the output. */
  url: string
}

/** The most recent ComfyUI run and its output files. */
export interface ComfyRun {
  promptId: string
  outputs: ComfyOutput[]
}

/** Absolute media directories of the open project, for sharing with ComfyUI. */
export interface ProjectMediaDirs {
  /** Where Inline Studio keeps imported inputs — point ComfyUI's --input-directory here. */
  inputDir: string
  /** Where Inline Studio keeps generated outputs — point ComfyUI's --output-directory here. */
  outputDir: string
}

/** Result of exporting a whole project to a portable .zip. */
export interface ProjectExportResult {
  /** Absolute path of the written .zip archive. */
  path: string
}

/** Summary of an "export frames to folder" run. */
export interface ExportResult {
  /** Absolute directory the files were written to. */
  dir: string
  /** Count of frame outputs exported. */
  exported: number
  /** Names of frames skipped because they had no Output yet. */
  skipped: string[]
}
