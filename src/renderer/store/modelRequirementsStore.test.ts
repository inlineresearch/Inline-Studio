import { describe, expect, it } from 'vitest'
import type { ModelRequirements } from '@shared/coreNodes'
import { activeDownload, type ComponentDownload } from './modelRequirementsStore'

const REQS: ModelRequirements = {
  allPresent: false,
  components: [
    {
      id: 'vae',
      label: 'VAE',
      category: 'vae',
      present: false,
      localPath: 'vae/x',
      repo: 'r',
      source: 'r/vae',
    },
    {
      id: 'diffusion',
      label: 'Diffusion model',
      category: 'diffusion_models',
      present: false,
      localPath: 'd',
      repo: 'r',
      source: 'r',
    },
  ],
}

describe('activeDownload', () => {
  it('returns null when nothing is downloading', () => {
    expect(activeDownload({}, REQS)).toBeNull()
  })

  it('surfaces an in-progress download with its component label', () => {
    const downloads: Record<string, ComponentDownload> = {
      vae: { fraction: 0.4, status: 'Downloading VAE…' },
    }
    expect(activeDownload(downloads, REQS)).toEqual({
      label: 'VAE',
      fraction: 0.4,
      status: 'Downloading VAE…',
    })
  })

  it('ignores errored downloads (they are not "active")', () => {
    const downloads: Record<string, ComponentDownload> = {
      vae: { fraction: 0.2, status: 'Failed', error: 'boom' },
    }
    expect(activeDownload(downloads, REQS)).toBeNull()
  })

  it('falls back to a generic label when the component is unknown', () => {
    const downloads: Record<string, ComponentDownload> = {
      mystery: { fraction: 0.1, status: 'Downloading…' },
    }
    expect(activeDownload(downloads, REQS)?.label).toBe('model')
  })
})
