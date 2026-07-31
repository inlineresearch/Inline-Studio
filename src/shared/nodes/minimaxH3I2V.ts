/**
 * fal.ai `minimax/h3/image-to-video` - MiniMax H3 (Hailuo 03) image-to-video, 2K.
 *
 * Two image ports: the first frame, and an optional last frame that turns the run into a
 * first-to-last keyframe interpolation. There is no `aspect_ratio` field here - H3 takes the output
 * framing from the start image. See https://fal.ai/models/minimax/h3/image-to-video.
 */
import { constantEndpoint } from './builders'
import { H3_BASE_PARAMS, buildH3Body, estimateH3Price } from './minimaxH3Shared'
import { portMedia, type NodeDef, type ResolvedInputs } from './types'

export const MINIMAX_H3_I2V: NodeDef = {
  id: 'minimax/h3/image-to-video',
  title: 'MiniMax H3 · Image → Video',
  category: 'Video',
  provider: 'fal',
  outputKind: 'video',
  inputs: [
    { id: 'image', label: 'Image', kind: 'image', required: true },
    { id: 'end_image', label: 'End image', kind: 'image', required: false },
  ],
  params: H3_BASE_PARAMS,
  outputs: [{ id: 'video', label: 'Video', kind: 'video' }],
  resolveEndpoint: constantEndpoint('minimax/h3/image-to-video'),
  buildRequest: (params, resolved: ResolvedInputs) => {
    const end = portMedia(MINIMAX_H3_I2V, resolved, 'end_image')[0]
    return buildH3Body(params, {
      image_url: portMedia(MINIMAX_H3_I2V, resolved, 'image')[0] ?? '',
      ...(end ? { end_image_url: end } : {}),
    })
  },
  estimatePrice: (params) => estimateH3Price(params),
}
