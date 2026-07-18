/**
 * fal.ai `bytedance/seedance-2.0/reference-to-video` — Seedance 2.0 reference-to-video: the prompt
 * references wired image/video/audio inputs (@Image1, @Video1, @Audio1). All inputs are optional.
 * See https://fal.ai/models/bytedance/seedance-2.0/reference-to-video.
 */
import { constantEndpoint } from './builders'
import { SEEDANCE_PARAMS, buildSeedanceBody, estimateSeedancePrice } from './seedanceShared'
import type { NodeDef, ResolvedInputs } from './types'

export const SEEDANCE_REF2V: NodeDef = {
  id: 'bytedance/seedance-2.0/reference-to-video',
  title: 'Seedance 2.0 · Reference → Video',
  category: 'Video',
  provider: 'fal',
  outputKind: 'video',
  inputs: [
    { id: 'image_urls', label: 'Images', kind: 'image[]', required: false },
    { id: 'video_urls', label: 'Video', kind: 'video', required: false },
    { id: 'audio_urls', label: 'Audio', kind: 'audio', required: false },
  ],
  params: SEEDANCE_PARAMS,
  outputs: [{ id: 'video', label: 'Video', kind: 'video' }],
  resolveEndpoint: constantEndpoint('bytedance/seedance-2.0/reference-to-video'),
  buildRequest: (params, resolved: ResolvedInputs) =>
    buildSeedanceBody(params, {
      ...(resolved.images.length > 0 ? { image_urls: resolved.images } : {}),
      ...(resolved.videos.length > 0 ? { video_urls: resolved.videos } : {}),
      ...(resolved.audios.length > 0 ? { audio_urls: resolved.audios } : {}),
    }),
  estimatePrice: (params) => estimateSeedancePrice(params),
}
