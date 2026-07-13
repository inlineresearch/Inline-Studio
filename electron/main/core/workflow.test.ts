import { describe, expect, it } from 'vitest'
import type { MoodboardConnector, MoodboardItem } from '@shared/types'
import { serializeWorkflow } from './workflow'

function coreItem(id: string, type: string, params: Record<string, unknown> = {}): MoodboardItem {
  return item(id, 'core', { core: { type, params } })
}

function promptItem(id: string, text: string): MoodboardItem {
  return item(id, 'prompt', { promptText: text })
}

function item(id: string, type: MoodboardItem['type'], data: MoodboardItem['data']): MoodboardItem {
  return {
    id,
    projectId: 'p',
    type,
    assetId: null,
    frameId: null,
    parentId: null,
    data,
    x: 0,
    y: 0,
    width: 0,
    height: 0,
    rotation: 0,
    zIndex: 0,
    createdAt: 0,
    updatedAt: 0,
  }
}

function edge(
  from: string,
  to: string,
  sourceHandle: string,
  targetHandle: string,
): MoodboardConnector {
  return {
    id: `${from}->${to}`,
    projectId: 'p',
    fromItemId: from,
    toItemId: to,
    label: null,
    data: { sourceHandle, targetHandle },
    createdAt: 0,
  }
}

describe('serializeWorkflow', () => {
  const items = [
    coreItem('m', 'load/diffusion-model', { file: 'z.safetensors' }),
    coreItem('te', 'load/text-encoder', { file: 'q' }),
    promptItem('p', 'a fox'),
    coreItem('enc', 'encode/text'),
    coreItem('lat', 'latent/empty'),
    coreItem('s', 'sample', { steps: 8 }),
    coreItem('v', 'load/vae', { file: 'ae' }),
    coreItem('d', 'vae/decode'),
    coreItem('orphan', 'load/vae'),
  ]
  const connectors = [
    edge('te', 'enc', 'text_encoder', 'text_encoder'),
    edge('p', 'enc', 'out', 'prompt'),
    edge('m', 's', 'model', 'model'),
    edge('enc', 's', 'conditioning', 'positive'),
    edge('lat', 's', 'latent', 'latent'),
    edge('v', 'd', 'vae', 'vae'),
    edge('s', 'd', 'latent', 'latent'),
  ]

  it('emits only the target closure', () => {
    const { graph, target } = serializeWorkflow('d', items, connectors, () => null)
    expect(target).toBe('d')
    const types = graph.nodes.map((n) => n.type).sort()
    expect(types).toEqual(
      [
        'encode/text',
        'input/text',
        'latent/empty',
        'load/diffusion-model',
        'load/text-encoder',
        'load/vae',
        'sample',
        'vae/decode',
      ].sort(),
    )
    expect(graph.nodes.find((n) => n.id === 'orphan')).toBeUndefined()
  })

  it('wires typed edges by Core port id', () => {
    const { graph } = serializeWorkflow('d', items, connectors, () => null)
    const sample = graph.nodes.find((n) => n.id === 's') as { inputs: Record<string, unknown> }
    expect(sample.inputs.model).toEqual({ from: 'm', output: 'model' })
    expect(sample.inputs.positive).toEqual({ from: 'enc', output: 'conditioning' })
    expect(sample.inputs.latent).toEqual({ from: 'lat', output: 'latent' })
  })

  it('maps a Prompt node to an input/text node with the port remapped to text', () => {
    const { graph } = serializeWorkflow('d', items, connectors, () => null)
    const prompt = graph.nodes.find((n) => n.id === 'p') as {
      type: string
      params: { text: string }
    }
    expect(prompt.type).toBe('input/text')
    expect(prompt.params.text).toBe('a fox')
    const enc = graph.nodes.find((n) => n.id === 'enc') as { inputs: Record<string, unknown> }
    expect(enc.inputs.prompt).toEqual({ from: 'p', output: 'text' })
  })
})
