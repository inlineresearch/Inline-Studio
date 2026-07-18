/**
 * fal.ai `krea/v2/large/text-to-image` - Krea v2 Large text-to-image.
 * See https://fal.ai/models/krea/v2/large/text-to-image.
 */
import { constantEndpoint, approxPrice, selectParam, seedParam, putSeed } from './builders'
import type { NodeDef } from './types'

const ASPECT_RATIOS = ['1:1', '4:3', '3:2', '16:9', '2.35:1', '4:5', '2:3', '9:16'] as const
const CREATIVITY = ['raw', 'low', 'medium', 'high'] as const

export const KREA_V2: NodeDef = {
  id: 'krea/v2/large/text-to-image',
  title: 'Krea v2 Large',
  category: 'Image',
  provider: 'fal',
  outputKind: 'image',
  inputs: [],
  params: [
    selectParam('aspect_ratio', 'Aspect ratio', ASPECT_RATIOS, '1:1'),
    selectParam('creativity', 'Creativity', CREATIVITY, 'medium'),
    seedParam(),
  ],
  outputs: [{ id: 'images', label: 'Image(s)', kind: 'image[]' }],
  resolveEndpoint: constantEndpoint('krea/v2/large/text-to-image'),
  buildRequest: (params) => {
    const body: Record<string, unknown> = {
      prompt: String(params.prompt ?? ''),
      aspect_ratio: params.aspect_ratio ?? '1:1',
      creativity: params.creativity ?? 'medium',
    }
    putSeed(body, params)
    return body
  },
  estimatePrice: () => approxPrice(0.06),
}
