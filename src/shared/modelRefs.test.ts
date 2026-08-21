import { describe, expect, it } from 'vitest'
import { coreNodeTypes, modelFilenames } from './modelRefs'

describe('model references in an exported graph', () => {
  it('finds the filenames a real exported node carries', () => {
    // Shape taken from a real project's moodboard_items row, not invented.
    const recipe = {
      graph: {
        items: [
          {
            id: '1',
            type: 'core',
            data: {
              core: {
                type: 'black-forest-labs/flux-2',
                params: {
                  model: 'flux-2-klein-4b.safetensors',
                  text_encoder: 'qwen_3_4b.safetensors',
                  vae: 'flux2-vae.safetensors',
                  steps: 20,
                  prompt: 'a cat',
                },
              },
            },
          },
          {
            id: '2',
            type: 'core',
            data: { core: { type: 'load/lora', params: { file: 'my-trained.safetensors' } } },
          },
        ],
      },
    }
    expect(modelFilenames(recipe)).toEqual([
      'flux-2-klein-4b.safetensors',
      'qwen_3_4b.safetensors',
      'flux2-vae.safetensors',
      'my-trained.safetensors',
    ])
  })

  it('ignores prose and dedupes a file two nodes share', () => {
    const graph = {
      a: { model: 'z_image_bf16.safetensors', prompt: 'a portrait, shot on film' },
      b: { model: 'z_image_bf16.safetensors', steps: 8 },
    }
    expect(modelFilenames(graph)).toEqual(['z_image_bf16.safetensors'])
  })

  it('takes the name a file lands under, not the path it was exported with', () => {
    expect(modelFilenames({ file: 'split_files/vae/ae.safetensors' })).toEqual(['ae.safetensors'])
  })

  it('finds nothing in a graph with no weights', () => {
    expect(modelFilenames({ items: [{ data: { text: 'hello' } }] })).toEqual([])
  })
})

describe('coreNodeTypes', () => {
  it('finds every Core node type in an exported graph, deduped', () => {
    const graph = {
      graph: {
        items: [
          { type: 'core', data: { core: { type: 'black-forest-labs/flux-2', params: {} } } },
          { type: 'core', data: { core: { type: 'load/vae', params: {} } } },
          { type: 'prompt', data: { promptText: 'x' } },
          { type: 'core', data: { core: { type: 'load/vae', params: {} } } },
        ],
      },
    }
    expect(coreNodeTypes(graph).sort()).toEqual(['black-forest-labs/flux-2', 'load/vae'])
  })

  it('is the only thing that can speak for a graph left on auto', () => {
    // Every loader on auto: the graph names no weight anywhere, so the filename sweep is empty and
    // a workflow needing several GB would report needing nothing.
    const graph = {
      graph: {
        items: [{ type: 'core', data: { core: { type: 'load/diffusion-model', params: {} } } }],
      },
    }
    expect(modelFilenames(graph)).toEqual([])
    expect(coreNodeTypes(graph)).toEqual(['load/diffusion-model'])
  })

  it('ignores anything that is not a core node', () => {
    expect(coreNodeTypes({ items: [{ type: 'loader', data: { assetIds: ['a'] } }] })).toEqual([])
    expect(coreNodeTypes(null)).toEqual([])
  })
})
