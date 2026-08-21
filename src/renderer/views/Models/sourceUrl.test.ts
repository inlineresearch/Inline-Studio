import { describe, expect, it } from 'vitest'
import type { RegistryModel } from '@shared/types'
import { sourceUrl } from './sourceUrl'

const base: RegistryModel = {
  id: 'x',
  label: 'x',
  filename: 'x.safetensors',
  category: 'vae',
  kind: 'hf_file',
  repo: 'Comfy-Org/z_image',
  path: 'split_files/vae/ae.safetensors',
  url: '',
  verified: true,
  optional: false,
  sizeBytes: null,
  updated: '',
  group: 'x',
  precision: '',
}

describe('a model source as an openable page', () => {
  it('points a file at its blob view, not the repo root', () => {
    expect(sourceUrl(base)).toBe(
      'https://huggingface.co/Comfy-Org/z_image/blob/main/split_files/vae/ae.safetensors',
    )
  })

  it('points a folder component at its tree view', () => {
    expect(sourceUrl({ ...base, kind: 'hf_folder', path: 'text_encoder' })).toBe(
      'https://huggingface.co/Comfy-Org/z_image/tree/main/text_encoder',
    )
  })

  it('uses a direct link as given', () => {
    expect(sourceUrl({ ...base, kind: 'url', url: 'https://example.com/m.safetensors' })).toBe(
      'https://example.com/m.safetensors',
    )
  })

  it('has no link when there is no repo, so the caller renders plain text', () => {
    expect(sourceUrl({ ...base, repo: '' })).toBe('')
  })
})
