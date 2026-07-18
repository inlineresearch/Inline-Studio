/**
 * The declarative contract for a generation node (a fal.ai model surfaced on the canvas).
 *
 * A `NodeDef` is pure data + pure functions — no Electron, no fs, no network. The renderer
 * uses it to render input ports + param widgets; the main-process executor uses it to build
 * the fal request and parse the response. Adding a model = one new file + a registry line.
 *
 * This file is imported by BOTH the renderer and main, so it must stay shell-agnostic
 * (open-core rule — see MEMORY: open-core-packages).
 */

/** The common input/output type system. Ports declare what kind of media flows through them. */
export type PortKind = 'image' | 'image[]' | 'video' | 'audio' | 'text' | 'path'

/** A typed input socket on a node (left side). */
export interface InputPort {
  id: string
  label: string
  kind: PortKind
  /** If required and unwired, the node can't run (the play button stays disabled). */
  required: boolean
}

/** A typed output socket on a node (right side). */
export interface OutputPort {
  id: string
  label: string
  kind: PortKind
}

/**
 * One user-editable parameter, rendered generically by GenNode as a widget.
 * `advanced: true` hides it behind the node's "More options" disclosure (only primary params
 * — those without the flag — show by default).
 */
export type ParamField =
  | { key: string; label: string; widget: 'text' | 'textarea'; default: string; advanced?: boolean }
  | {
      key: string
      label: string
      widget: 'select'
      options: { value: string; label: string }[]
      default: string
      advanced?: boolean
    }
  | {
      key: string
      label: string
      widget: 'number'
      default: number
      min?: number
      max?: number
      step?: number
      advanced?: boolean
    }
  | { key: string; label: string; widget: 'boolean'; default: boolean; advanced?: boolean }

/** Concrete param values for a node instance (persisted on its backing Frame). */
export type ParamValues = Record<string, string | number | boolean>

/**
 * A node's inputs after the executor has resolved + uploaded them to fal storage. Grouped by
 * kind so `buildRequest` can map them to the model's specific fields (e.g. `image_url` vs
 * `image_urls`). Empty arrays when nothing is wired to that kind of port.
 */
export interface ResolvedInputs {
  images: string[]
  masks: string[]
  videos: string[]
  audios: string[]
  texts: string[]
}

/** An estimated cost for one generation. Models price differently (per image / MP / second / …). */
export interface PriceEstimate {
  /** Estimated total USD for one run with the current params. */
  amount: number
  /** Always an estimate (fal prices can change; some models bill on real output size). */
  approx?: boolean
}

/**
 * Compact USD label for a price estimate, e.g. `~$0.145`. Sub-dollar costs use 3 decimals so
 * models that differ only in the third place (e.g. $0.145 vs $0.136) read distinctly; a dollar or
 * more rounds to cents.
 */
export function formatPrice(est: PriceEstimate): string {
  const s = `$${est.amount < 1 ? est.amount.toFixed(3) : est.amount.toFixed(2)}`
  return est.approx ? `~${s}` : s
}

/**
 * A generation node definition. Its functions carry ALL model-specific knowledge and are
 * pure/unit-testable; the executor stays model-agnostic.
 *
 * Note there is deliberately no `parseOutputs` here: the browser builds the request
 * (`resolveEndpoint` + `buildRequest`) and hands it to Core, which then submits, polls, and
 * downloads the result asynchronously — the browser is never in the loop for the response. Parsing
 * therefore lives in Core (`inline_core/studio/fal.py: parse_outputs`), keyed on `outputKind`.
 */
export interface NodeDef {
  /** The fal model id, e.g. `openai/gpt-image-2`. Also the registry key + `Frame.modelId`. */
  id: string
  /** Display title, e.g. `GPT Image 2`. */
  title: string
  /** Grouping label for the node palette, e.g. `Image` / `Video`. */
  category: string
  provider: 'fal'
  /** The kind of media this node produces → the backing `Frame.kind`. */
  outputKind: 'image' | 'video' | 'audio'
  /**
   * When true, the node runs without a connected Prompt node — the model derives its own prompt
   * from its media inputs (e.g. Sonilo reads the video), and a wired prompt only steers the result.
   * Default (absent) keeps the prompt required, which is the norm.
   */
  promptOptional?: boolean
  inputs: InputPort[]
  params: ParamField[]
  outputs: OutputPort[]
  /** Pick the fal endpoint from the resolved inputs (e.g. text-to-image vs image-to-image). PURE. */
  resolveEndpoint(resolved: ResolvedInputs): string
  /**
   * Build the fal request body from the frame's persisted param values + resolved (uploaded) input
   * URLs. Params arrive as `unknown` (JSON from the DB), so implementations coerce defensively. PURE.
   */
  buildRequest(params: Record<string, unknown>, resolved: ResolvedInputs): Record<string, unknown>
  /** Estimate the USD cost of one generation from the current param values. PURE; optional. */
  estimatePrice?(params: Record<string, unknown>): PriceEstimate | null
}

/** Fill a param-value map from a node def's field defaults (used when creating a fal frame). */
export function defaultParams(def: NodeDef): ParamValues {
  const out: ParamValues = {}
  for (const field of def.params) {
    out[field.key] = field.default
  }
  return out
}

/** An empty `ResolvedInputs` (all kinds empty) — a convenience for callers/tests. */
export function emptyResolvedInputs(): ResolvedInputs {
  return { images: [], masks: [], videos: [], audios: [], texts: [] }
}
