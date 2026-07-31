/**
 * fal.ai FLUX.2 - text-to-image and multi-reference editing, hosted.
 * See https://fal.ai/models/fal-ai/flux-2 and https://fal.ai/models/fal-ai/flux-2/edit.
 *
 * The same family the local `black-forest-labs/flux-2` Core node runs, without the ~16 GB of
 * weights. The edit endpoint takes an **array** of reference images and the prompt addresses them
 * by position ("the jacket from image 2"), so it maps onto the same `image[]` port and the numbered
 * reference strip the Core node uses.
 */
import {
  approxPrice,
  constantEndpoint,
  numberParam,
  selectParam,
  seedParam,
  putSeed,
} from './builders'
import type { NodeDef, ResolvedInputs } from './types'

/** fal's named sizes; the API also accepts an explicit {width,height} we do not expose. */
const IMAGE_SIZES = [
  'square_hd',
  'square',
  'portrait_4_3',
  'portrait_16_9',
  'landscape_4_3',
  'landscape_16_9',
] as const
const OUTPUT_FORMATS = ['jpeg', 'png', 'webp'] as const
/** Trades a little fidelity for latency; `none` is the reference path. */
const ACCELERATION = ['none', 'regular', 'high'] as const

/** Params shared by both endpoints, in the order they matter to a user. */
function sharedParams() {
  return [
    selectParam('image_size', 'Image size', IMAGE_SIZES, 'landscape_4_3'),
    numberParam('num_inference_steps', 'Steps', 28, { min: 4, max: 50, step: 1 }),
    numberParam('guidance_scale', 'Guidance', 2.5, { min: 0, max: 20, step: 0.5 }),
    numberParam('num_images', 'Count', 1, { min: 1, max: 4, step: 1 }),
    selectParam('acceleration', 'Acceleration', ACCELERATION, 'none'),
    selectParam('output_format', 'Output format', OUTPUT_FORMATS, 'png'),
    seedParam(),
  ]
}

function sharedBody(params: Record<string, unknown>): Record<string, unknown> {
  const body: Record<string, unknown> = {
    prompt: String(params.prompt ?? ''),
    image_size: params.image_size ?? 'landscape_4_3',
    num_inference_steps: Number(params.num_inference_steps ?? 28),
    guidance_scale: Number(params.guidance_scale ?? 2.5),
    num_images: Number(params.num_images ?? 1),
    acceleration: params.acceleration ?? 'none',
    output_format: params.output_format ?? 'png',
  }
  putSeed(body, params)
  return body
}

const perImage = (params: Record<string, unknown>, unit: number) =>
  approxPrice(unit * Math.max(1, Number(params.num_images ?? 1)))

export const FLUX_2: NodeDef = {
  id: 'fal-ai/flux-2',
  title: 'FLUX.2',
  category: 'Image',
  provider: 'fal',
  outputKind: 'image',
  inputs: [],
  params: sharedParams(),
  outputs: [{ id: 'images', label: 'Image(s)', kind: 'image[]' }],
  resolveEndpoint: constantEndpoint('fal-ai/flux-2'),
  buildRequest: (params) => sharedBody(params),
  estimatePrice: (params) => perImage(params, 0.04),
}

export const FLUX_2_EDIT: NodeDef = {
  id: 'fal-ai/flux-2/edit',
  title: 'FLUX.2 · Edit',
  category: 'Image',
  provider: 'fal',
  outputKind: 'image',
  // A list input: one reference edits that image, several compose from them. Order is the order
  // the prompt refers to them in.
  inputs: [{ id: 'image_urls', label: 'Reference image(s)', kind: 'image[]', required: true }],
  params: sharedParams(),
  outputs: [{ id: 'images', label: 'Image(s)', kind: 'image[]' }],
  resolveEndpoint: constantEndpoint('fal-ai/flux-2/edit'),
  buildRequest: (params, resolved: ResolvedInputs) => ({
    ...sharedBody(params),
    image_urls: resolved.images,
  }),
  estimatePrice: (params) => perImage(params, 0.04),
}
