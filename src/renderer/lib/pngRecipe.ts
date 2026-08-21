/**
 * Read the Inline recipe embedded in a PNG (a tEXt/iTXt chunk keyed `inline-studio`, written by Core
 * at generation time). Lets a generated image - dropped from the Outputs tab or shared and imported
 * from disk - rebuild the graph that made it. Pure byte parsing, no dependencies.
 */

/** One canvas node in a recipe (trimmed by Core to the rebuild-relevant fields). */
export interface RecipeItem {
  id: string
  type: string
  data: Record<string, unknown>
  x: number
  y: number
  width?: number
  height?: number
  assetId?: string | null
  frameId?: string | null
}

export interface RecipeConnector {
  fromItemId: string
  toItemId: string
  data: Record<string, unknown>
}

/** 2 typed every param and moved each node's models beside it; 1 was a flat param map. */
export const RECIPE_VERSION = 2

/** What a param is, so a reader never has to guess it from the param's name. */
export type RecipeParamType =
  | 'string'
  | 'text'
  | 'number'
  | 'boolean'
  | 'enum'
  | 'seed'
  | 'model'
  | 'character'
  | 'file'

export interface RecipeParam {
  type: RecipeParamType
  value: unknown
}

/** Where a model comes from: its models/ folder and a direct link. Mirrors Comfy's node models. */
export interface RecipeModel {
  directory: string
  name: string
  url: string
}

export interface Recipe {
  version?: number
  app?: string
  target?: string
  coreType?: string
  params?: Record<string, unknown>
  prompt?: string
  graph?: { items: RecipeItem[]; connectors: RecipeConnector[] }
}

const PNG_SIG = [137, 80, 78, 71, 13, 10, 26, 10]
const KEYWORD = 'inline-studio'

const latin1 = (b: Uint8Array, from: number, to: number): string => {
  let s = ''
  for (let i = from; i < to; i++) s += String.fromCharCode(b[i])
  return s
}

/** The text of the `inline-studio` chunk, or null. Handles tEXt and uncompressed iTXt (Core writes
 * ASCII-escaped JSON, so it lands as tEXt; iTXt is handled for robustness). */
function extractRecipeText(buf: Uint8Array): string | null {
  if (buf.length < 8 || PNG_SIG.some((b, i) => buf[i] !== b)) return null
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
  let off = 8
  while (off + 8 <= buf.length) {
    const len = dv.getUint32(off)
    const type = latin1(buf, off + 4, off + 8)
    const start = off + 8
    const end = start + len
    if (end > buf.length) break
    if (type === 'tEXt') {
      let z = start
      while (z < end && buf[z] !== 0) z++
      if (latin1(buf, start, z) === KEYWORD) return latin1(buf, z + 1, end)
    } else if (type === 'iTXt') {
      let z = start
      while (z < end && buf[z] !== 0) z++
      const key = latin1(buf, start, z)
      z++ // keyword null
      const compressed = buf[z] !== 0
      z += 2 // compression flag + method
      while (z < end && buf[z] !== 0) z++ // language tag
      z++
      while (z < end && buf[z] !== 0) z++ // translated keyword
      z++
      if (key === KEYWORD && !compressed) {
        return new TextDecoder().decode(buf.subarray(z, end))
      }
    }
    if (type === 'IEND') break
    off = end + 4 // skip data + CRC
  }
  return null
}

/** Parse a recipe from raw PNG bytes, or null if none/invalid. */
export function parseRecipeBytes(bytes: Uint8Array): Recipe | null {
  const text = extractRecipeText(bytes)
  if (!text) return null
  try {
    const parsed: unknown = JSON.parse(text)
    if (parsed && typeof parsed === 'object' && (parsed as Recipe).app === 'inline-studio') {
      return parsed as Recipe
    }
  } catch {
    return null
  }
  return null
}

/** Read a recipe from an image Blob/File (fetched output or a dropped PNG). */
export async function readRecipeFromBlob(blob: Blob): Promise<Recipe | null> {
  try {
    return parseRecipeBytes(new Uint8Array(await blob.arrayBuffer()))
  } catch {
    return null
  }
}

/** Parse an exported `.inline-graph.json`. Same shape and same `app` guard as the PNG chunk, so a
 * graph round-trips whether it was shared as an image or as a file. */
export function parseRecipeJson(text: string): Recipe | null {
  try {
    const parsed: unknown = JSON.parse(text)
    if (parsed && typeof parsed === 'object' && (parsed as Recipe).app === 'inline-studio') {
      return parsed as Recipe
    }
  } catch {
    return null
  }
  return null
}

/** Read a recipe from a dropped JSON File. */
export async function readRecipeFromJsonFile(file: Blob): Promise<Recipe | null> {
  try {
    return parseRecipeJson(await file.text())
  } catch {
    return null
  }
}

/**
 * A node's params as plain values, from either shape.
 *
 * v2 types every param as {type, value}; v1 stored the bare value. Files already exported carry
 * v1, and an image someone generated last week has to keep rebuilding its graph.
 */
export function paramValues(params: unknown): Record<string, unknown> {
  if (!params || typeof params !== 'object') return {}
  const out: Record<string, unknown> = {}
  for (const [key, entry] of Object.entries(params as Record<string, unknown>)) {
    const typed =
      entry && typeof entry === 'object' && 'value' in (entry as Record<string, unknown>)
    out[key] = typed ? (entry as { value: unknown }).value : entry
  }
  return out
}
