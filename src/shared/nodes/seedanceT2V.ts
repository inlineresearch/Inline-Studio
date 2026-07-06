/**
 * fal.ai `bytedance/seedance-2.0/text-to-video` — ByteDance Seedance 2.0 text-to-video.
 * See https://fal.ai/models/bytedance/seedance-2.0/text-to-video.
 */
import { constantEndpoint, parseSingleVideo } from './builders'
import { SEEDANCE_PARAMS, buildSeedanceBody, estimateSeedancePrice } from './seedanceShared'
import type { NodeDef } from './types'

export const SEEDANCE_T2V: NodeDef = {
  id: 'bytedance/seedance-2.0/text-to-video',
  title: 'Seedance 2.0 · Text → Video',
  category: 'Video',
  provider: 'fal',
  outputKind: 'video',
  inputs: [],
  params: SEEDANCE_PARAMS,
  outputs: [{ id: 'video', label: 'Video', kind: 'video' }],
  resolveEndpoint: constantEndpoint('bytedance/seedance-2.0/text-to-video'),
  buildRequest: (params) => buildSeedanceBody(params, {}),
  parseOutputs: (response) => parseSingleVideo(response),
  estimatePrice: (params) => estimateSeedancePrice(params),
}
