/**
 * fal.ai `fal-ai/nano-banana-pro/edit` - image editing (image-to-image) with Nano Banana Pro.
 * See https://fal.ai/models/fal-ai/nano-banana-pro/edit.
 */
import { constantEndpoint, approxPrice, selectParam, numberParam } from './builders'
import { withCharacterPrompt, withCharacterRefs, type NodeDef, type ResolvedInputs } from './types'

// fal documents no ceiling; Google documents 14 references, of which 5 is the identity band.
const MAX_IMAGES = 14
const MAX_CHARACTER_IMAGES = 5

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
const RESOLUTIONS = ['1K', '2K', '4K'] as const

export const NANO_BANANA_PRO: NodeDef = {
  id: 'fal-ai/nano-banana-pro/edit',
  title: 'Nano Banana Pro · Edit',
  category: 'Image',
  provider: 'fal',
  outputKind: 'image',
  inputs: [
    { id: 'image_urls', label: 'Image(s)', kind: 'image[]', required: true },
    { id: 'character', label: 'Character', kind: 'character', required: false },
  ],
  character: {
    port: 'image_urls',
    style: 'ordinal',
    maxImages: MAX_IMAGES,
    maxRefs: MAX_CHARACTER_IMAGES,
  },
  params: [
    selectParam('aspect_ratio', 'Aspect ratio', ASPECT_RATIOS, 'auto'),
    selectParam('resolution', 'Resolution', RESOLUTIONS, '1K'),
    numberParam('num_images', 'Count', 1, { min: 1, max: 4, step: 1 }),
  ],
  outputs: [{ id: 'images', label: 'Image(s)', kind: 'image[]' }],
  resolveEndpoint: constantEndpoint('fal-ai/nano-banana-pro/edit'),
  buildRequest: (params, resolved: ResolvedInputs) => ({
    prompt: withCharacterPrompt(resolved, String(params.prompt ?? '')),
    image_urls: withCharacterRefs(NANO_BANANA_PRO, resolved, 'image_urls'),
    aspect_ratio: params.aspect_ratio ?? 'auto',
    resolution: params.resolution ?? '1K',
    num_images: Number(params.num_images ?? 1),
  }),
  // $0.15 per image; 4K outputs bill at 2×.
  estimatePrice: (params) => {
    const mult = String(params.resolution ?? '1K') === '4K' ? 2 : 1
    const count = Math.max(1, Number(params.num_images ?? 1))
    return approxPrice(0.15 * mult * count)
  },
}
