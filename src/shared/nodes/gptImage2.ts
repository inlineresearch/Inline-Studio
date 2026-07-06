/**
 * fal.ai `openai/gpt-image-2` — text-to-image, and image-to-image when image inputs are wired.
 * See https://fal.ai/models/openai/gpt-image-2.
 */
import { type NodeDef, type ResolvedInputs } from './types'
import {
  constantEndpoint,
  parseImageArray,
  approxPrice,
  selectParam,
  numberParam,
} from './builders'

const SIZE_PRESETS = [
  'square_hd',
  'square',
  'portrait_4_3',
  'portrait_16_9',
  'landscape_4_3',
  'landscape_16_9',
] as const

// OpenAI's models live under the `openai/` owner on fal — NOT the `fal-ai/` namespace (which is
// only for fal's own models). Prefixing with fal-ai/ makes the queue 404 ("Application openai not found").
//
// There is a SINGLE endpoint: passing `image_urls` switches it to image-editing mode. There is no
// `/image-to-image` sub-path — submitting to one 404s on the queue result fetch.
// See https://fal.ai/models/openai/gpt-image-2/api.
const ENDPOINT = 'openai/gpt-image-2'

export const GPT_IMAGE_2: NodeDef = {
  id: 'openai/gpt-image-2',
  title: 'GPT Image 2',
  category: 'Image',
  provider: 'fal',
  outputKind: 'image',
  inputs: [
    // Wiring any image here switches the node to image-to-image (editing).
    { id: 'images', label: 'Image(s)', kind: 'image[]', required: false },
    { id: 'mask', label: 'Mask', kind: 'image', required: false },
  ],
  // NOTE: `prompt` is not a param — it's fed via the node's prompt input (a connected Prompt node).
  params: [
    selectParam('image_size', 'Size', SIZE_PRESETS, 'landscape_4_3'),
    selectParam('quality', 'Quality', ['low', 'medium', 'high'], 'high'),
    numberParam('num_images', 'Count', 1, { min: 1, max: 8, step: 1 }),
    selectParam('output_format', 'Format', ['png', 'jpeg', 'webp'], 'png'),
  ],
  outputs: [{ id: 'images', label: 'Image(s)', kind: 'image[]' }],

  // One endpoint for both text-to-image and editing; `image_urls` in the body selects the mode.
  resolveEndpoint: constantEndpoint(ENDPOINT),

  buildRequest(params: Record<string, unknown>, resolved: ResolvedInputs): Record<string, unknown> {
    const body: Record<string, unknown> = {
      prompt: String(params.prompt ?? ''),
      image_size: params.image_size ?? 'landscape_4_3',
      quality: params.quality ?? 'high',
      num_images: Number(params.num_images ?? 1),
      output_format: params.output_format ?? 'png',
    }
    if (resolved.images.length > 0) {
      body.image_urls = resolved.images
      if (resolved.masks.length > 0) body.mask_url = resolved.masks[0]
    }
    return body
  },

  parseOutputs: (response) => parseImageArray(response),

  // fal bills per image by resolution × quality. We use the ~1MP (1024×768) tier as the baseline
  // (most presets are ~0.5–1MP), times the image count. Source: fal.ai/models/openai/gpt-image-2.
  estimatePrice(params) {
    const perImage: Record<string, number> = { low: 0.005, medium: 0.037, high: 0.145 }
    const rate = perImage[String(params.quality ?? 'high')] ?? perImage.high
    const count = Math.max(1, Number(params.num_images ?? 1))
    return approxPrice(rate * count)
  },
}
