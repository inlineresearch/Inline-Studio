/**
 * Build an Inline Core graph (schemaVersion 1) from a frame: the model node plus the source nodes
 * for its prompt and image inputs. The client owns the closure boundary (Inline Core contract):
 * curated inputs enter as input/* source nodes, so nothing upstream is recomputed.
 */
import { join } from 'node:path'
import { getFalFrame, frameInputAssetPaths } from '../frames/store'
import { promptTextForFrame } from '../moodboard/store'
import { getOpenProjectFolder } from '../db'

export interface CoreGraph {
  schemaVersion: number
  nodes: Array<Record<string, unknown>>
}

export function buildFrameGraph(frameId: string): { graph: CoreGraph; target: string } {
  const { provider, modelId, params } = getFalFrame(frameId)
  if (provider !== 'core' || !modelId) throw new Error('This is not an Inline Core node.')
  const folder = getOpenProjectFolder()
  if (!folder) throw new Error('No project is open.')

  const nodes: Array<Record<string, unknown>> = []
  const inputs: Record<string, unknown> = {}

  const promptText = promptTextForFrame(frameId)
  if (promptText) {
    const id = `${frameId}::prompt`
    nodes.push({ id, type: 'input/text', params: { text: promptText } })
    inputs.prompt = { from: id, output: 'text' }
  }

  frameInputAssetPaths(frameId).forEach((input, index) => {
    const id = `${frameId}::img${index}`
    nodes.push({
      id,
      type: 'input/image',
      params: { asset: { ref: 'path', path: join(folder, input.filePath) } },
    })
    // Single image port for now; multi-input routing is a later refinement.
    inputs.image = { from: id, output: 'image' }
  })

  nodes.push({ id: frameId, type: modelId, params, inputs })
  return { graph: { schemaVersion: 1, nodes }, target: frameId }
}
