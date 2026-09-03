/**
 * Mapping between a node def's input ports and the canvas handle ids its dots render with.
 *
 * The canvas has always named a Generate node's dots `in` (media) and `audio`, and every existing
 * project's connectors store those strings - renaming them would orphan the edges on the canvas. So
 * the first image/video port and the first audio port keep those legacy ids; any further port uses
 * its own id. What gets persisted as an input's `handle` is always the real port id.
 */
import { mediaFamily, type InputPort, type NodeDef } from './types'

/** Ports share a dot-naming group when they draw from the same media bucket - except audio, which
 *  has always had its own dot, so it groups on its own. */
function dotGroup(port: InputPort): 'media' | 'audio' | 'character' | null {
  if (port.kind === 'character') return 'character'
  const family = mediaFamily(port.kind)
  if (family === null) return null
  return family === 'audio' ? 'audio' : 'media'
}

/** The ports a Generate node renders dots for, in top-to-bottom order (media, audio, character). */
export function dotPorts(def: NodeDef): InputPort[] {
  return [
    ...def.inputs.filter((p) => dotGroup(p) === 'media'),
    ...def.inputs.filter((p) => dotGroup(p) === 'audio'),
    ...def.inputs.filter((p) => dotGroup(p) === 'character'),
  ]
}

/** The canvas handle id a port's dot renders with. */
export function handleIdForPort(def: NodeDef, port: InputPort): string {
  const group = dotGroup(port)
  if (group === null) return port.id
  // Character came after the legacy ids, so it never inherits one - its handle is always its id.
  if (group === 'character') return port.id
  const peers = def.inputs.filter((p) => dotGroup(p) === group)
  return peers[0] === port ? (group === 'audio' ? 'audio' : 'in') : port.id
}

/** The port id a canvas handle refers to, or null when the handle isn't one of this def's inputs. */
export function portIdForHandle(def: NodeDef, handleId: string | null | undefined): string | null {
  if (!handleId) return null
  for (const port of def.inputs) {
    if (handleIdForPort(def, port) === handleId) return port.id
  }
  return null
}
