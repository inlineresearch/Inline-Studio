import { describe, expect, it } from 'vitest'
import type { NodeDescriptor } from '@shared/coreNodes'
import { missingInputs, missingInputsMessage } from './missingInputs'

function descriptor(params: NodeDescriptor['params']): NodeDescriptor {
  return {
    type: 'alibaba/z-image-turbo',
    title: 'Z-Image',
    category: 'Generate',
    icon: 'wand',
    inputs: [],
    outputs: [],
    params,
    source: 'core',
    outputKind: 'image',
  } as unknown as NodeDescriptor
}

const LORA_FIELD = {
  key: 'file',
  label: 'LoRA',
  widget: 'select' as const,
  default: '',
  optionsFrom: 'loras',
  options: [
    { value: '', label: 'Auto' },
    { value: 'installed.safetensors', label: 'installed.safetensors' },
  ],
}

describe('missingInputs', () => {
  it('flags a pick the catalog does not have', () => {
    const missing = missingInputs(descriptor([LORA_FIELD]), { file: 'from-elsewhere.safetensors' })
    expect(missing).toEqual([{ key: 'file', label: 'LoRA', value: 'from-elsewhere.safetensors' }])
  })

  it('accepts a pick that is installed', () => {
    expect(missingInputs(descriptor([LORA_FIELD]), { file: 'installed.safetensors' })).toEqual([])
  })

  it('treats an empty pick as fine, since Core auto-resolves it', () => {
    expect(missingInputs(descriptor([LORA_FIELD]), { file: '' })).toEqual([])
    expect(missingInputs(descriptor([LORA_FIELD]), {})).toEqual([])
  })

  it('says nothing while the descriptor is still loading', () => {
    // Otherwise every node on the canvas would flash red on a slow registry fetch.
    expect(missingInputs(undefined, { file: 'x.safetensors' })).toEqual([])
  })

  it('says nothing when the catalog list is empty', () => {
    // An empty option list means the scan has not landed, not that the file is absent.
    const empty = { ...LORA_FIELD, options: [] }
    expect(missingInputs(descriptor([empty]), { file: 'x.safetensors' })).toEqual([])
  })

  it('ignores params with no catalog behind them', () => {
    const steps = { key: 'steps', label: 'Steps', widget: 'number' as const, default: 8 }
    expect(missingInputs(descriptor([steps]), { steps: 999 })).toEqual([])
  })

  it('reports every missing pick on the node', () => {
    const vae = { ...LORA_FIELD, key: 'vae', label: 'VAE', optionsFrom: 'vae' }
    const missing = missingInputs(descriptor([LORA_FIELD, vae]), {
      file: 'a.safetensors',
      vae: 'b.safetensors',
    })
    expect(missing.map((m) => m.key)).toEqual(['file', 'vae'])
  })
})

describe('missingInputsMessage', () => {
  it('is empty when nothing is missing', () => {
    expect(missingInputsMessage([])).toBe('')
  })

  it('names each missing file so the user knows what to install', () => {
    const message = missingInputsMessage([
      { key: 'file', label: 'LoRA', value: 'a.safetensors' },
      { key: 'vae', label: 'VAE', value: 'b.safetensors' },
    ])
    expect(message).toContain('LoRA: a.safetensors')
    expect(message).toContain('VAE: b.safetensors')
  })
})
