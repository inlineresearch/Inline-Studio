/**
 * fal.ai `fal-ai/ltx-2.3-quality/image-to-video` — animates a single input image into an MP4.
 * See https://fal.ai/models/fal-ai/ltx-2.3-quality/image-to-video.
 *
 * This is the video counterpart to GPT Image 2: it proves the registry generalizes across I/O
 * types (image input → video output) and across run durations (video generation is long-running).
 */
import { type NodeDef, type ResolvedInputs } from './types'
import {
  constantEndpoint,
  approxPrice,
  numberParam,
  booleanParam,
  seedParam,
  putSeed,
} from './builders'

const ENDPOINT = 'fal-ai/ltx-2.3-quality/image-to-video'

/**
 * fal's `resolution` field takes a named aspect preset (or an ImageSize object) — a raw "W x H"
 * string 422s. We use the presets (fal maps them to known-good, correctly-divisible dimensions);
 * `width`/`height` are fal's standard preset pixel sizes, used only for the cost estimate.
 */
const RESOLUTIONS = [
  { value: 'landscape_16_9', label: 'Landscape 16:9', width: 1024, height: 576 },
  { value: 'portrait_16_9', label: 'Portrait 9:16', width: 576, height: 1024 },
  { value: 'square_hd', label: 'Square HD', width: 1024, height: 1024 },
  { value: 'landscape_4_3', label: 'Landscape 4:3', width: 1024, height: 768 },
  { value: 'portrait_4_3', label: 'Portrait 3:4', width: 768, height: 1024 },
  { value: 'square', label: 'Square', width: 512, height: 512 },
  { value: 'auto', label: 'Auto (match input)', width: 1024, height: 576 },
] as const

const DEFAULT_RESOLUTION = 'landscape_16_9'

/** Coerce a stored value to a valid preset, migrating legacy "W x H" strings by orientation. */
function toResolutionPreset(value: unknown): (typeof RESOLUTIONS)[number] {
  const byValue = (name: string): (typeof RESOLUTIONS)[number] =>
    RESOLUTIONS.find((r) => r.value === name) ?? RESOLUTIONS[0]
  const v = String(value ?? '')
  if (RESOLUTIONS.some((r) => r.value === v)) return byValue(v)
  const [w, h] = v.split(/x/i).map((s) => Number(s.trim()))
  if (w && h) return byValue(w === h ? 'square_hd' : w > h ? 'landscape_16_9' : 'portrait_16_9')
  return byValue(DEFAULT_RESOLUTION)
}

export const LTX_I2V: NodeDef = {
  id: 'fal-ai/ltx-2.3-quality/image-to-video',
  title: 'LTX 2.3 · Image → Video',
  category: 'Video',
  provider: 'fal',
  outputKind: 'video',
  inputs: [{ id: 'image', label: 'Image', kind: 'image', required: true }],
  // NOTE: `prompt` is not a param — it's fed via the node's prompt input (a connected Prompt node).
  params: [
    numberParam('number_of_frames', 'Frames', 96, { min: 1, max: 300, step: 1 }),
    // Custom labels (not value === label), so declared inline rather than via selectParam.
    {
      key: 'resolution',
      label: 'Resolution',
      widget: 'select',
      options: RESOLUTIONS.map((r) => ({ value: r.value, label: r.label })),
      default: DEFAULT_RESOLUTION,
      advanced: true,
    },
    booleanParam('generate_audio', 'Generate audio', false),
    seedParam(),
  ],
  outputs: [{ id: 'video', label: 'Video', kind: 'video' }],

  resolveEndpoint: constantEndpoint(ENDPOINT),

  buildRequest(params: Record<string, unknown>, resolved: ResolvedInputs): Record<string, unknown> {
    const body: Record<string, unknown> = {
      prompt: String(params.prompt ?? ''),
      image_url: resolved.images[0] ?? '',
      number_of_frames: Number(params.number_of_frames ?? 96),
      resolution: toResolutionPreset(params.resolution).value,
      generate_audio: Boolean(params.generate_audio ?? false),
    }
    putSeed(body, params)
    return body
  },

  // fal bills $0.0024075 per megapixel of generated video, where MP = width × height × frames.
  // Source: fal.ai/models/fal-ai/ltx-2.3-quality/image-to-video.
  estimatePrice(params) {
    const { width, height } = toResolutionPreset(params.resolution)
    const frames = Number(params.number_of_frames ?? 96)
    if (!frames) return null
    const megapixels = (width * height * frames) / 1_000_000
    return approxPrice(megapixels * 0.0024075)
  },
}
