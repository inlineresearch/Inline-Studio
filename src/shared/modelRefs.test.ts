import { describe, expect, it } from 'vitest'
import { modelFilenames } from './modelRefs'

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
