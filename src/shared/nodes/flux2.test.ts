import { describe, expect, it } from 'vitest'

import { FLUX_2, FLUX_2_EDIT } from './flux2'
import { getNodeDef, listNodeDefs } from './registry'
import type { ResolvedInputs } from './types'

const RESOLVED: ResolvedInputs = {
  images: ['https://a/1.png', 'https://a/2.png'],
} as ResolvedInputs

describe('fal FLUX.2 nodes', () => {
  it('are registered and findable by their fal endpoint id', () => {
    expect(getNodeDef('fal-ai/flux-2')).toBe(FLUX_2)
    expect(getNodeDef('fal-ai/flux-2/edit')).toBe(FLUX_2_EDIT)
    expect(listNodeDefs()).toContain(FLUX_2_EDIT)
  })

  it('takes references as an ordered list, since the prompt addresses them by position', () => {
    expect(FLUX_2_EDIT.inputs[0]).toMatchObject({ kind: 'image[]', required: true })
    const body = FLUX_2_EDIT.buildRequest({ prompt: 'the fox from image 1' }, RESOLVED)
    // Order preserved: "image 1" must be the first wire.
    expect(body.image_urls).toEqual(['https://a/1.png', 'https://a/2.png'])
  })

  it('text-to-image takes no inputs and sends no image_urls', () => {
    expect(FLUX_2.inputs).toEqual([])
    const body = FLUX_2.buildRequest({ prompt: 'a fox' }, {} as ResolvedInputs)
    expect(body).not.toHaveProperty('image_urls')
    expect(body.prompt).toBe('a fox')
  })

  it('sends the schema defaults when the user has not touched a param', () => {
    // These mirror fal's own defaults; drifting from them changes output for no stated reason.
    const body = FLUX_2.buildRequest({ prompt: 'x' }, {} as ResolvedInputs)
    expect(body).toMatchObject({
      image_size: 'landscape_4_3',
      num_inference_steps: 28,
      guidance_scale: 2.5,
      num_images: 1,
      acceleration: 'none',
      output_format: 'png',
    })
  })

  it('passes explicit params through and stays inside the API bounds', () => {
    const body = FLUX_2.buildRequest(
      { prompt: 'x', num_inference_steps: 50, guidance_scale: 0, num_images: 4 },
      {} as ResolvedInputs,
    )
    expect(body).toMatchObject({ num_inference_steps: 50, guidance_scale: 0, num_images: 4 })
    const steps = FLUX_2.params.find((p) => p.key === 'num_inference_steps')
    expect(steps).toMatchObject({ min: 4, max: 50 })
    const guidance = FLUX_2.params.find((p) => p.key === 'guidance_scale')
    expect(guidance).toMatchObject({ min: 0, max: 20 })
  })

  it('omits the seed unless one is set, so runs stay random by default', () => {
    expect(FLUX_2.buildRequest({ prompt: 'x' }, {} as ResolvedInputs)).not.toHaveProperty('seed')
    expect(FLUX_2.buildRequest({ prompt: 'x', seed: 7 }, {} as ResolvedInputs).seed).toBe(7)
  })

  it('prices per image', () => {
    const one = FLUX_2.estimatePrice?.({ num_images: 1 })
    const four = FLUX_2.estimatePrice?.({ num_images: 4 })
    expect(four?.amount).toBeCloseTo((one?.amount ?? 0) * 4, 5)
  })
})
