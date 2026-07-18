/**
 * fal.ai `fal-ai/nano-banana-2` — text-to-image (Google's Nano Banana 2, served via fal).
 * See https://fal.ai/models/fal-ai/nano-banana-2.
 */
import { constantEndpoint, approxPrice, selectParam, numberParam } from './builders'
import type { NodeDef } from './types'

const ASPECT_RATIOS = [
  'auto',
  '21:9',
  '16:9',
  '3:2',
  '4:3',
  '5:4',
  '1:1',
  '4:5',
  '3:4',
  '2:3',
  '9:16',
] as const
const RESOLUTIONS = ['1K', '2K', '4K', '512x512'] as const
const RES_MULT: Record<string, number> = { '1K': 1, '2K': 1.5, '4K': 2, '512x512': 0.75 }

export const NANO_BANANA_2: NodeDef = {
  id: 'fal-ai/nano-banana-2',
  title: 'Nano Banana 2',
  category: 'Image',
  provider: 'fal',
  outputKind: 'image',
  inputs: [],
  params: [
    selectParam('aspect_ratio', 'Aspect ratio', ASPECT_RATIOS, 'auto'),
    selectParam('resolution', 'Resolution', RESOLUTIONS, '1K'),
    numberParam('limit_generations', 'Count', 1, { min: 1, max: 4, step: 1 }),
  ],
  outputs: [{ id: 'images', label: 'Image(s)', kind: 'image[]' }],
  resolveEndpoint: constantEndpoint('fal-ai/nano-banana-2'),
  buildRequest: (params) => ({
    prompt: String(params.prompt ?? ''),
    aspect_ratio: params.aspect_ratio ?? 'auto',
    resolution: params.resolution ?? '1K',
    limit_generations: Number(params.limit_generations ?? 1),
  }),
  // $0.08 per image at 1K, scaled by resolution tier × count.
  estimatePrice: (params) => {
    const mult = RES_MULT[String(params.resolution ?? '1K')] ?? 1
    const count = Math.max(1, Number(params.limit_generations ?? 1))
    return approxPrice(0.08 * mult * count)
  },
}
