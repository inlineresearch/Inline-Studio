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
   * Which generation engine backs this frame. `fal` = a declarative fal.ai node (see
   * `modelId`/`params`). `core` = an Inline Core node. `unset` = a legacy frame with no engine
   * (renders as a plain viewer). (Legacy projects may still carry a `'comfy'` string from before
   * ComfyUI was removed; it renders as a plain viewer too.)
   */
  provider: 'fal' | 'core' | 'unset'
  /** For `provider:'fal'`: the registry model id (e.g. `openai/gpt-image-2`). */
  modelId: string | null
  /** For `provider:'fal'`: the node's editable param values (keyed by the NodeDef param keys). */
  params: Record<string, unknown>
  /** Workflow template this frame generates with, if chosen. */
  workflowTemplateId: string | null
  createdAt: number
  updatedAt: number
}

/**
 * One of a frame's inputs (a frame can have several). An input is either a library
 * asset (`assetId`) or a live link to another frame's output (`sourceFrameId`) - the
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
  /** A "Load Assets" node: holds library asset refs in `data.assetIds`, feeds its hero downstream. */
  | 'loader'
  /** A "Control Space" node: a 3D pose editor whose rendered OpenPose map (`data.controlAssetId`)
   * feeds a gen node's control input. */
  | 'controlSpace'
  /** Trainer-canvas: picks a training dataset and feeds it downstream. */
  | 'trainDataset'
  /** Trainer-canvas: auto-captions a dataset's images. */
  | 'caption'
  /** Trainer-canvas: runs a LoRA training job (run/stop/resume, live steps + logs). */
  | 'trainer'
  /** Trainer-canvas: plots a training run's loss curve. */
  | 'lossGraph'
  /** Utility: read-only host telemetry (CPU/RAM/VRAM). No handles; usable on either canvas. */
  | 'resource'

/**
 * Which canvas an item lives on. The Studio moodboard and the Trainer tab's graph share the same
 * item/connector tables, kept apart by this discriminator.
 */
export type CanvasSurface = 'studio' | 'trainer'

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
 * are sequential (seconds); nothing here is persisted - it's recomputed from the wired
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
  /** Optional http(s) URL - renders the text as a clickable link. */
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
  /** `loader` ("Load Assets") node: ordered library asset ids it holds; the first (hero) feeds
   * downstream. A pure viewer with no generation. (Legacy: also flags an old frame-based loader.) */
  loader?: boolean
  /** `loader` node: the library asset ids it holds, in order (hero = first). */
  assetIds?: string[]
  /** `controlSpace` node: the library asset id of the last rendered OpenPose control map, fed
   * downstream into a gen node's control input. */
  controlAssetId?: string
  /** `controlSpace` node: the serialized 3D scene so the editor reopens on the saved pose.
   * `characters` is the current multi-character form; `joints` is the legacy single-character field,
   * still read for scenes saved before multi-character support. */
  controlScene?: {
    joints?: [number, number, number][]
    characters?: { joints: [number, number, number][] }[]
    fov?: number
    /** Camera pose so a saved render reproduces the exact framing on reopen. */
    camera?: { position: [number, number, number]; target: [number, number, number] }
    /** Which control map the node last rendered: an OpenPose skeleton or a depth map. */
    output?: 'pose' | 'depth'
    /** Output aspect (w/h) of the rendered map; must match the gen node's resolution or the pose
     * comes out stretched. Defaults to 1 (square). */
    aspect?: number
    /** How each character was turned relative to the camera when the map was rendered. */
    facing?: ('front' | 'three-quarter' | 'profile' | 'back' | null)[]
    /** Prompt text the facing needs, resolved at render time and appended to a downstream gen node's
     * prompt / negative prompt. A control map cannot express "facing away" - only the text encoder
     * can - so Control Space states it here. Null (or absent) = nothing to add. */
    promptHint?: { positive: string; negative: string } | null
    /** Whether the facing prompt hint is applied downstream (default true). */
    applyPromptHint?: boolean
  }
  /** `trainDataset` / `caption` / `trainer` nodes: the training dataset they operate on. */
  datasetId?: string | null
  /** `caption` node: re-caption images that already have a caption. */
  overwrite?: boolean
  /** `trainer` / `lossGraph` nodes: the training run they're bound to. Persisted so the node
   * rebinds after a reload and can still offer Resume on an interrupted run. */
  runId?: string | null
  /** `trainer` node: its hyperparameters (edited in the Adjust sidebar, off the node face). */
  hyperparams?: Partial<TrainingHyperparams>
  /** Core graph node: the Inline Core node type + its param values (see coreNodes.ts). */
  core?: {
    type: string
    params: Record<string, unknown>
    /** The active media this node produced - shown large and flowed downstream. */
    output?: CoreTakeRef
    /** Recent renders (newest first); `output` points at the active one. Drives the take-history
     * strip on generation nodes. */
    outputs?: CoreTakeRef[]
  }
}

/** One render in a Core node's history: the media file plus the recipe fields that let switching to
 * it restore the node's settings and its prompt (the reproducible-recipe feature). */
export interface CoreTakeRef {
  takeId: string
  filePath: string
  kind: 'image' | 'video' | 'audio'
  /** Stamped at generation (ms); absent on renders made before it was tracked. */
  createdAt?: number
  /** The settings this take was generated with, so switching to it restores the node's params. */
  params?: Record<string, unknown>
  /** The prompt this take used (from the upstream prompt node), for display + restore. */
  prompt?: string
}

export interface MoodboardItem {
  id: string
  projectId: string
  /** Which canvas this item belongs to (defaults to the Studio moodboard). */
  surface: CanvasSurface
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
  /** Derived from the nodes it joins - a connector never spans two canvases. */
  surface: CanvasSurface
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

// LoRA training ------------------------------------------------------------

/** Turbo needs a training adapter to avoid "turbo drift"; a de-turbo base trains without one. */
/** The model family a LoRA is trained for. */
export type TrainingArch = 'z-image' | 'krea2'

/**
 * Which base checkpoint a run trains against. `raw` is Krea 2's undistilled base (the recommended
 * path - the LoRA then applies to Turbo at generation time); the turbo modes train against a
 * distilled checkpoint and need a training adapter to avoid turbo drift.
 */
export type TrainingBaseMode = 'turbo_adapter' | 'deturbo' | 'raw'

/**
 * Precision of the frozen base during training. The LoRA itself is always full precision, so `nf4`
 * only costs base fidelity - and it is what puts Krea 2 (a 26GB base) on a 16GB card. `auto` sizes
 * the base plus its activations against your GPU and picks for you.
 */
export type TrainingBaseQuant = 'auto' | 'none' | 'nf4'

/** Stream saved activations to host RAM so a bf16 base fits a smaller card. */
export type TrainingOffload = 'auto' | 'on' | 'off'

/**
 * Which Linears the adapter attaches to. `full` adapts the feed-forward and projection layers as
 * well, which is stronger on short style runs; `attention` is the Krea 2 authors' advice for long
 * runs, where adapting everything starts to cost prompt adherence.
 */
export type TrainingLoraScope = 'full' | 'attention'

export type TrainingStatus =
  | 'queued'
  | 'training'
  | 'done'
  | 'failed'
  | 'cancelled'
  /** Died to a crash/OOM/restart; resumable from its last checkpoint. */
  | 'interrupted'

/** A named set of images (+ captions) that trains one LoRA. */
export interface TrainingDataset {
  id: string
  projectId: string
  name: string
  /** Token injected into every caption (the character/style trigger); may be empty. */
  triggerWord: string
  createdAt: number
  updatedAt: number
}

/** One image in a dataset, with its caption. Backed by a library asset. */
export interface TrainingDatasetItem {
  id: string
  datasetId: string
  assetId: string
  caption: string
  position: number
  createdAt: number
}

/** A local auto-caption model the Trainer can offer. Defined in Core's `training/caption.py`. */
export interface CaptionerModel {
  id: string
  label: string
  repo: string
}

/** LoRA training hyperparameters (persisted as JSON on a run). */
export interface TrainingHyperparams {
  /** Defaults to `z-image` when absent, so runs saved before Krea 2 keep working. */
  arch?: TrainingArch
  baseMode: TrainingBaseMode
  /** Defaults to `auto`. Only Krea 2 has a 4-bit path; Z-Image always trains in bf16. */
  baseQuant?: TrainingBaseQuant
  /**
   * Defaults to `auto`. Offloads saved activations to host RAM during the step so a full-precision
   * base fits a card that could not otherwise hold base + activations. `auto` turns it on only when
   * the bf16 plan would not fit resident. Trades some speed for the VRAM.
   */
  offload?: TrainingOffload
  /** Defaults to `full`. */
  loraScope?: TrainingLoraScope
  /** 0 to 1: how often a step trains on an empty caption, so the LoRA holds without the trigger. */
  captionDropout?: number
  /** Mirror every image, doubling the dataset. Wrong for text or anything asymmetric. */
  flipAugment?: boolean
  /** LoRA rank (dim). */
  rank: number
  /** LoRA alpha; defaults to `rank`. */
  alpha: number
  learningRate: number
  steps: number
  batchSize: number
  /** Square training resolution in px (e.g. 1024). */
  resolution: number
  /** Checkpoint every N steps. */
  saveEvery: number
  /** GPU indices to train on; `[]` = auto (first available). */
  gpuIds: number[]
  /** Filename (no extension) for the produced LoRA. Blank = derived from the run name. */
  outputName?: string
}

/** A training run over a dataset, producing a LoRA `.safetensors`. */
export interface TrainingRun {
  id: string
  projectId: string
  datasetId: string
  name: string
  status: TrainingStatus
  hyperparams: TrainingHyperparams
  /** `models/loras/`-relative path of the produced LoRA, once done. */
  outputLoraPath: string | null
  progressFraction: number
  progressStatus: string | null
  step: number
  totalSteps: number
  error: string | null
  createdAt: number
  updatedAt: number
}

/** Main → renderer: training progress for a run (0..1) + step + latest loss (for the loss curve). */
export interface TrainingProgressEvent {
  runId: string
  fraction: number
  step: number
  totalSteps: number
  loss?: number
  status?: string
}

/** Main → renderer: a mid-training sample preview image was written (project-relative path). */
export interface TrainingSampleEvent {
  runId: string
  step: number
  path: string
}

/** Main → renderer: auto-caption progress for a dataset (one tick per captioned image). */
export interface CaptionProgressEvent {
  datasetId: string
  /** Images captioned so far, of `total` queued this run. */
  done: number
  total: number
  /** The item just captioned, if this tick is for one. */
  itemId: string | null
  /** True on the final tick, when the captioner has exited. */
  finished: boolean
}

/** Main → renderer: one stdout line from the trainer subprocess (shown in the Trainer node). */
export interface TrainingLogEvent {
  runId: string
  line: string
}

/** Main → renderer: a training run finished; `outputLoraPath` is the produced LoRA. */
export interface TrainingDoneEvent {
  runId: string
  outputLoraPath: string
}

/** Main → renderer: a training run failed. */
export interface TrainingErrorEvent {
  runId: string
  error: string
}

/** Per-GPU live stats (NVML). */
export interface GpuStat {
  index: number
  name: string
  /** GPU utilization %, 0..100. */
  utilization: number
  /** VRAM used / total, in bytes. */
  memoryUsed: number
  memoryTotal: number
  /** Degrees C, or null if unavailable. */
  temperature: number | null
}

/** Main → renderer: periodic host + GPU telemetry (Trainer tab). */
export interface SystemStatsEvent {
  /** CPU utilization %, 0..100. */
  cpu: number
  /** System RAM used / total, in bytes. */
  ramUsed: number
  ramTotal: number
  gpus: GpuStat[]
}

/** Main → renderer: extension install progress. Payload shapes live in `extensions.ts`. */
export type {
  InstallProgressEvent as ExtensionInstallProgressEvent,
  InstallSuccess as ExtensionInstallDoneEvent,
  InstallFailure as ExtensionInstallErrorEvent,
} from './extensions'

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

/** Result of pinging the configured Inline Core engine. */
export interface CoreStatus {
  running: boolean
  url: string
}

/** Absolute media directories of the open project. */
export interface ProjectMediaDirs {
  /** Where Inline Studio keeps imported inputs - point ComfyUI's --input-directory here. */
  inputDir: string
  /** Where Inline Studio keeps generated outputs - point ComfyUI's --output-directory here. */
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
