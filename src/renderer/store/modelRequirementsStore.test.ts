import { describe, expect, it } from 'vitest'
import type { InlineStudioApi } from '@shared/ipc'
import { ok } from '@shared/result'
import { setStudioClient } from '../lib/studio'
import type { ModelRequirements } from '@shared/coreNodes'
import {
  activeDownload,
  useModelRequirementsStore,
  type ComponentDownload,
} from './modelRequirementsStore'
import { useCoreNodesStore } from './coreNodesStore'

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

describe('a keyed entry', () => {
  // Train LoRA's components follow the architecture in its own settings, so two of them on
  // different archs would overwrite each other's answer under a plain node-type key.
  it('caches under the key, not the node type', async () => {
    const asked: { nodeType: string; params?: Record<string, unknown> }[] = []
    setStudioClient({
      models: {
        requirements: async (nodeType: string, params?: Record<string, unknown>) => {
          asked.push({ nodeType, params })
          return ok(REQS)
        },
      },
    } as unknown as InlineStudioApi)

    const store = useModelRequirementsStore.getState()
    await store.load('train/lora', { hyperparams: { arch: 'krea2' } }, 'train/lora:krea2')
    await store.load('train/lora', { hyperparams: { arch: 'flux2' } }, 'train/lora:flux2')

    const state = useModelRequirementsStore.getState()
    expect(Object.keys(state.byType).sort()).toEqual(['train/lora:flux2', 'train/lora:krea2'])
    expect(state.asked['train/lora:krea2']).toEqual({
      nodeType: 'train/lora',
      params: { hyperparams: { arch: 'krea2' } },
    })
    expect(asked.map((a) => a.nodeType)).toEqual(['train/lora', 'train/lora'])
  })

  it('receives download events reported against its node type', () => {
    useModelRequirementsStore.setState({
      asked: {
        'train/lora:krea2': { nodeType: 'train/lora' },
        'train/lora:flux2': { nodeType: 'train/lora' },
      },
      downloads: {},
    })
    useModelRequirementsStore
      .getState()
      .onProgress({ nodeType: 'train/lora', componentId: 'vae', fraction: 0.5, status: 'x' })

    const downloads = useModelRequirementsStore.getState().downloads
    expect(downloads['train/lora:krea2']?.vae?.fraction).toBe(0.5)
    expect(downloads['train/lora:flux2']?.vae?.fraction).toBe(0.5)
  })
})

describe('load caching', () => {
  const stub = (): { nodeType: string; params?: Record<string, unknown> }[] => {
    const asked: { nodeType: string; params?: Record<string, unknown> }[] = []
    setStudioClient({
      models: {
        requirements: async (nodeType: string, params?: Record<string, unknown>) => {
          asked.push({ nodeType, params })
          return ok(REQS)
        },
      },
    } as unknown as InlineStudioApi)
    return asked
  }

  const reset = (): void =>
    useModelRequirementsStore.setState({ byType: {}, asked: {}, loadedFor: {} })

  it('asks Core once for a type it has already answered', async () => {
    reset()
    const asked = stub()
    const store = useModelRequirementsStore.getState()
    // The canvas remounts a node whenever it scrolls back into view, so this is the common case.
    await store.load('zimage')
    await store.load('zimage')
    await store.load('zimage')
    expect(asked).toHaveLength(1)
  })

  it('coalesces concurrent asks about one type into a single call', async () => {
    reset()
    const asked = stub()
    const store = useModelRequirementsStore.getState()
    await Promise.all([store.load('zimage'), store.load('zimage'), store.load('zimage')])
    expect(asked).toHaveLength(1)
  })

  it('re-asks when refresh is set, which is how a finished download refreshes', async () => {
    reset()
    const asked = stub()
    const store = useModelRequirementsStore.getState()
    await store.load('zimage')
    await store.load('zimage', undefined, undefined, true)
    expect(asked).toHaveLength(2)
  })

  it('re-asks when the registry moves, so a dropped-in file is picked up', async () => {
    reset()
    const asked = stub()
    const store = useModelRequirementsStore.getState()
    await store.load('zimage')
    useCoreNodesStore.setState({ registryVersion: 'v2' })
    await store.load('zimage')
    expect(asked).toHaveLength(2)
  })

  it('re-asks when the same key is given different params', async () => {
    reset()
    const asked = stub()
    const store = useModelRequirementsStore.getState()
    await store.load('train/lora', { arch: 'krea2' }, 'train/lora')
    await store.load('train/lora', { arch: 'flux2' }, 'train/lora')
    expect(asked).toHaveLength(2)
  })
})
