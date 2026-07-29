/**
 * The Inline Core node descriptor contract, mirrored on the client. Inline Core serves these at
 * `GET /v1/models`; the canvas renders any node generically from its descriptor, so adding a node
 * type is a Core change with no Inline Studio release. Kept shell-agnostic (imported by renderer + main).
 *
 * Media kinds cross the wire as takes/assets; engine kinds (`model`, `latent`, ...) are opaque
 * handles passed between low-level nodes and are never a take. Only nodes with an `outputKind`
 * (media output) become Frames with take history; the rest are ephemeral plumbing nodes.
 */

export type PortKind =
  | 'image'
  | 'image[]'
  | 'video'
  | 'audio'
  | 'text'
  | 'mask'
  | 'model'
  | 'vae'
  | 'text-encoder'
  | 'lora'
  | 'conditioning'
  | 'latent'
  | 'control'

export interface CorePort {
  id: string
  label: string
  kind: PortKind
  required: boolean
}

export interface CoreParamOption {
  value: string
  label: string
}

export interface CoreParamField {
  key: string
  label: string
  widget: 'text' | 'textarea' | 'number' | 'boolean' | 'select' | 'seed'
  default: string | number | boolean
  min?: number
  max?: number
  step?: number
  options?: CoreParamOption[]
  /** A dynamic catalog Core fills from what is installed (checkpoints, loras, vae, ...). */
  optionsFrom?: string
  advanced?: boolean
}

export interface NodeDescriptor {
  type: string
  title: string
  category: string
  icon: string
  source: string
  /** Media output kind for a Frame node; null for plumbing nodes (engine outputs only). */
  outputKind: 'image' | 'video' | 'audio' | null
  inputs: CorePort[]
  outputs: CorePort[]
  params: CoreParamField[]
  /**
   * Internal building blocks (loaders, samplers, VAE, source inputs) - served for
   * validation/execution but never offered in the add-node menu. Keeps generation one-click: the
   * user sees only high-level model nodes (e.g. Z-Image Turbo). See `addableCoreNodes`.
   */
  hidden?: boolean
}

/** Descriptors the add-node menu may offer - everything not marked `hidden` by Core. */
export function addableCoreNodes(descriptors: NodeDescriptor[]): NodeDescriptor[] {
  return descriptors.filter((d) => !d.hidden)
}

/** The `GET /v1/models` payload. */
export interface CoreModels {
  registryVersion: string
  models: NodeDescriptor[]
}

/**
 * One model a node needs, whether it's on disk, and where it lives - the data behind a node's
 * "missing models" popup. Nothing is auto-downloaded: a component is `present` only when the user
 * placed files under `models/` or downloaded it from the popup (which writes into `models/`).
 */
export interface ModelComponent {
  id: string
  label: string
  /** The `models/` subfolder it belongs to (e.g. `diffusion_models`, `vae`, `text_encoders`). */
  category: string
  present: boolean
  /** Where it's expected / would land, relative to the models root. */
  localPath: string
  /** The Hugging Face repo the popup downloads it from. */
  repo: string
  /** "Which model" - the repo narrowed to the exact subfolder(s), e.g. `Owner/Repo/vae`. */
  source: string
  /** A suggested (not required) component - e.g. the opt-in ControlNet. Never counts toward
   * `allPresent`; the UI offers it as a separate download suggestion. */
  optional?: boolean
}

/** A node's model requirements with live presence (from `models:requirements`). */
export interface ModelRequirements {
  components: ModelComponent[]
  allPresent: boolean
}

const ENGINE_KINDS: readonly PortKind[] = [
  'model',
  'vae',
  'text-encoder',
  'lora',
  'conditioning',
  'latent',
]

/** Engine handles are opaque objects passed between low-level nodes; never a take. */
export function isEngineKind(kind: PortKind): boolean {
  return ENGINE_KINDS.includes(kind)
}

const MODEL_KINDS: readonly PortKind[] = ['model', 'vae', 'text-encoder', 'lora']

/**
 * A "model-family" handle - the loader plumbing (diffusion model, VAE, text encoder) that threads a
 * component into a node. On the canvas these dots extension to the **bottom** edge of a node while the
 * content/signal dots (image, latent, conditioning, …) extension to the **top**, so model wiring reads as
 * one band along the bottom and the actual image flow runs across the top.
 */
export function isModelPort(kind: PortKind): boolean {
  return MODEL_KINDS.includes(kind)
}

/** A node that emits media (has an `outputKind`) is a Frame with take history. */
export function producesFrame(descriptor: NodeDescriptor): boolean {
  return descriptor.outputKind !== null
}

/** Whether an output of `source` kind may feed an input of `target` kind. Mirrors Core. */
export function portsSatisfy(source: PortKind, target: PortKind): boolean {
  if (source === target) return true
  if (source === 'image' && target === 'image[]') return true
  // A control input accepts any image output (a raw image, a preprocessed control map, or a
  // Control Space render); the control map is just an image.
  return source === 'image' && target === 'control'
}

const PORT_COLORS: Record<PortKind, string> = {
  image: '#34d399',
  'image[]': '#34d399',
  video: '#22d3ee',
  audio: '#a78bfa',
  text: '#fbbf24',
  mask: '#f472b6',
  model: '#f87171',
  vae: '#fb923c',
  'text-encoder': '#facc15',
  lora: '#2dd4bf',
  conditioning: '#c084fc',
  latent: '#60a5fa',
  control: '#f9a8d4',
}

/** A stable color per port kind, for Comfy-style colored sockets and edges. */
export function portKindColor(kind: PortKind): string {
  return PORT_COLORS[kind] ?? '#a1a1aa'
}
