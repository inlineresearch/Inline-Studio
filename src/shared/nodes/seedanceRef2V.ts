/**
 * fal.ai `bytedance/seedance-2.0/reference-to-video` - Seedance 2.0 reference-to-video: the prompt
 * references wired image/video/audio inputs (@Image1, @Video1, @Audio1). All inputs are optional.
 * See https://fal.ai/models/bytedance/seedance-2.0/reference-to-video.
 */
import { constantEndpoint } from './builders'
import { SEEDANCE_PARAMS, buildSeedanceBody, estimateSeedancePrice } from './seedanceShared'
import {
  portMedia,
  withCharacterPrompt,
  withCharacterRefs,
  type NodeDef,
  type ResolvedInputs,
} from './types'

// fal's caps, trimmed client-side rather than 422'ing at the API.
const MAX_IMAGES = 9
const MAX_VIDEOS = 3
const MAX_AUDIOS = 3
const MAX_TOTAL = 12

export const SEEDANCE_REF2V: NodeDef = {
  id: 'bytedance/seedance-2.0/reference-to-video',
  title: 'Seedance 2.0 · Reference → Video',
  category: 'Video',
  provider: 'fal',
  outputKind: 'video',
  inputs: [
    { id: 'image_urls', label: 'Images', kind: 'image[]', required: false },
    // Lists, because fal takes 3 of each here and the prompt numbers them (@Video1, @Video2).
    { id: 'video_urls', label: 'Video', kind: 'video[]', required: false },
    { id: 'audio_urls', label: 'Audio', kind: 'audio[]', required: false },
    { id: 'character', label: 'Character', kind: 'character', required: false },
  ],
  character: {
    port: 'image_urls',
    style: 'at-image',
    maxImages: MAX_IMAGES,
    maxRefs: MAX_IMAGES,
    // ByteDance's partner validation refuses any reference carrying a face, so a character's
    // identity references cannot reach this endpoint at all - measured, not inferred: a wardrobe-only
    // payload of the same shape and prompt style is accepted, a face-only one is refused. Dropping
    // them here turns a two-minute wait and a charge into nothing being sent that cannot land.
    excludeRoles: ['face'],
    // Once the face references are gone the remaining images are an outfit and a build, so the
    // prompt has to say that rather than claim they show the character.
    roleLines: true,
  },
  params: SEEDANCE_PARAMS,
  outputs: [{ id: 'video', label: 'Video', kind: 'video' }],
  resolveEndpoint: constantEndpoint('bytedance/seedance-2.0/reference-to-video'),
  buildRequest: (params, resolved: ResolvedInputs) => {
    // Audio yields first when the combined cap bites: it is the input the model can do without.
    const images = withCharacterRefs(SEEDANCE_REF2V, resolved, 'image_urls').slice(0, MAX_IMAGES)
    const videos = portMedia(SEEDANCE_REF2V, resolved, 'video_urls').slice(0, MAX_VIDEOS)
    const audios = portMedia(SEEDANCE_REF2V, resolved, 'audio_urls')
      .slice(0, MAX_AUDIOS)
      .slice(0, Math.max(0, MAX_TOTAL - images.length - videos.length))
    return buildSeedanceBody(
      { ...params, prompt: withCharacterPrompt(resolved, String(params.prompt ?? '')) },
      {
        ...(images.length > 0 ? { image_urls: images } : {}),
        ...(videos.length > 0 ? { video_urls: videos } : {}),
        ...(audios.length > 0 ? { audio_urls: audios } : {}),
      },
    )
  },
  estimatePrice: (params) => estimateSeedancePrice(params),
}
