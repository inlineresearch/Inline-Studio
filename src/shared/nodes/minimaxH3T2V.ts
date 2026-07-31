/**
 * fal.ai `minimax/h3/text-to-video` - MiniMax H3 (Hailuo 03) text-to-video, 2K.
 * See https://fal.ai/models/minimax/h3/text-to-video.
 */
import { constantEndpoint, selectParam } from './builders'
import { H3_ASPECT_RATIOS, H3_BASE_PARAMS, buildH3Body, estimateH3Price } from './minimaxH3Shared'
import type { NodeDef } from './types'

export const MINIMAX_H3_T2V: NodeDef = {
  id: 'minimax/h3/text-to-video',
  title: 'MiniMax H3 · Text → Video',
  category: 'Video',
  provider: 'fal',
  outputKind: 'video',
  inputs: [],
  params: [
    ...H3_BASE_PARAMS,
    selectParam('aspect_ratio', 'Aspect ratio', H3_ASPECT_RATIOS, '16:9'),
  ],
  outputs: [{ id: 'video', label: 'Video', kind: 'video' }],
  resolveEndpoint: constantEndpoint('minimax/h3/text-to-video'),
  buildRequest: (params) =>
    buildH3Body(params, { aspect_ratio: String(params.aspect_ratio ?? '16:9') }),
  estimatePrice: (params) => estimateH3Price(params),
}
