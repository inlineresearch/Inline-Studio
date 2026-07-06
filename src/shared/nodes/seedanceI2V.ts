/**
 * fal.ai `bytedance/seedance-2.0/image-to-video` — ByteDance Seedance 2.0 image-to-video.
 * See https://fal.ai/models/bytedance/seedance-2.0/image-to-video.
 */
import { constantEndpoint, parseSingleVideo } from './builders'
import { SEEDANCE_PARAMS, buildSeedanceBody, estimateSeedancePrice } from './seedanceShared'
import type { NodeDef, ResolvedInputs } from './types'

export const SEEDANCE_I2V: NodeDef = {
  id: 'bytedance/seedance-2.0/image-to-video',
  title: 'Seedance 2.0 · Image → Video',
  category: 'Video',
  provider: 'fal',
  outputKind: 'video',
  inputs: [{ id: 'image', label: 'Image', kind: 'image', required: true }],
  params: SEEDANCE_PARAMS,
  outputs: [{ id: 'video', label: 'Video', kind: 'video' }],
  resolveEndpoint: constantEndpoint('bytedance/seedance-2.0/image-to-video'),
  buildRequest: (params, resolved: ResolvedInputs) =>
    buildSeedanceBody(params, { image_url: resolved.images[0] ?? '' }),
  parseOutputs: (response) => parseSingleVideo(response),
  estimatePrice: (params) => estimateSeedancePrice(params),
}
