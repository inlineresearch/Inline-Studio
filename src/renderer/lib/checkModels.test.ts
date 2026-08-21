import { describe, expect, it } from 'vitest'
import { graphModelRequests, type DeclaredComponent } from './checkModels'

/** FLUX.2 as Core declares it: one default set, the other builds offered as alternatives. */
const FLUX2: DeclaredComponent[] = [
  {
    localPath: 'diffusion_models/flux-2-klein-4b.safetensors',
    category: 'diffusion_models',
    present: false,
  },
  { localPath: 'text_encoders/qwen_3_4b.safetensors', category: 'text_encoders', present: false },
  { localPath: 'vae/flux2-vae.safetensors', category: 'vae', present: false },
  {
    localPath: 'diffusion_models/flux-2-klein-9b.safetensors',
    category: 'diffusion_models',
    present: false,
    optional: true,
  },
  {
    localPath: 'diffusion_models/flux2_dev_fp8mixed.safetensors',
    category: 'diffusion_models',
    present: false,
    optional: true,
  },
]

describe('graphModelRequests', () => {
  it('reports the build the workflow names, not the node default', () => {
    // The point of the rule: a graph tagged klein-9b must not be told klein-4b is missing.
    const wanted = graphModelRequests(['flux-2-klein-9b.safetensors'], { flux2: FLUX2 })
    const names = wanted.map((w) => w.filename)
    expect(names).toContain('flux-2-klein-9b.safetensors')
    expect(names).not.toContain('flux-2-klein-4b.safetensors')
  })

  it('still asks for the categories the workflow says nothing about', () => {
    const names = graphModelRequests(['flux-2-klein-9b.safetensors'], { flux2: FLUX2 }).map(
      (w) => w.filename,
    )
    expect(names).toContain('qwen_3_4b.safetensors')
    expect(names).toContain('flux2-vae.safetensors')
  })

  it('falls back to the whole default set when the graph names nothing', () => {
    // An export made before params were recorded names no weight at all.
    const names = graphModelRequests([], { flux2: FLUX2 }).map((w) => w.filename)
    expect(names.sort()).toEqual([
      'flux-2-klein-4b.safetensors',
      'flux2-vae.safetensors',
      'qwen_3_4b.safetensors',
    ])
  })

  it('never asks for an alternative build on its own', () => {
    expect(graphModelRequests([], { flux2: FLUX2 }).map((w) => w.filename)).not.toContain(
      'flux2_dev_fp8mixed.safetensors',
    )
  })

  it('skips what is already on disk', () => {
    const present = FLUX2.map((c) => (c.category === 'vae' ? { ...c, present: true } : c))
    const names = graphModelRequests([], { flux2: present }).map((w) => w.filename)
    expect(names).not.toContain('flux2-vae.safetensors')
  })

  it('keeps a named file once, whatever declares it', () => {
    const names = graphModelRequests(['flux-2-klein-4b.safetensors'], {
      a: FLUX2,
      b: FLUX2,
    }).map((w) => w.filename)
    expect(names.filter((n) => n === 'flux-2-klein-4b.safetensors')).toHaveLength(1)
  })
})
