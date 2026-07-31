/**
 * Resolve the images wired into a Core node's list input, in wiring order.
 *
 * FLUX.2 addresses reference images by position - "the jacket from image 2" - so the order the user
 * wired them in is meaning, not decoration. The node face numbers them so the prompt can name them,
 * which means this must produce exactly what `graph_build.py` sends to the engine: connector order,
 * and a Load Assets node contributing every asset it holds rather than only its first.
 *
 * Pure: the caller passes the stores' data in, so this is testable without React.
 */
import type { Asset, Frame, MoodboardConnector, MoodboardItem, Take } from '@shared/types'

export interface CoreInputThumb {
  /** 1-based, matching how a prompt refers to it ("image 1"). */
  index: number
  /** Project-relative media path, still to be run through `resolveMedia`. */
  filePath: string
  /** The canvas item this came from, for the hover title. */
  sourceId: string
  label: string
}

export interface ThumbContext {
  items: MoodboardItem[]
  connectors: MoodboardConnector[]
  assets: Asset[]
  frames: Frame[]
  takesByFrame: Record<string, Take[]>
}

/** Every image feeding `handle` on `itemId`, numbered from 1 in wiring order. */
export function resolveCoreInputThumbs(
  itemId: string,
  handle: string,
  ctx: ThumbContext,
): CoreInputThumb[] {
  const wired = ctx.connectors.filter(
    (c) => c.toItemId === itemId && (c.data?.targetHandle ?? 'in') === handle,
  )
  const thumbs: CoreInputThumb[] = []
  for (const connector of wired) {
    const source = ctx.items.find((i) => i.id === connector.fromItemId)
    if (!source) continue
    for (const filePath of sourcePaths(source, ctx)) {
      thumbs.push({
        index: thumbs.length + 1,
        filePath,
        sourceId: source.id,
        label: sourceLabel(source, ctx),
      })
    }
  }
  return thumbs
}

/**
 * The image(s) one canvas item contributes. A Load Assets node contributes all of them, matching
 * the engine's fan-out; every other kind contributes at most one.
 */
function sourcePaths(item: MoodboardItem, ctx: ThumbContext): string[] {
  switch (item.type) {
    case 'loader': {
      const ids = item.data.assetIds ?? []
      return ids.map((id) => assetPath(id, ctx)).filter(isPresent)
    }
    case 'controlSpace': {
      const path = item.data.controlAssetId ? assetPath(item.data.controlAssetId, ctx) : undefined
      return path ? [path] : []
    }
    case 'asset': {
      const path = item.assetId ? assetPath(item.assetId, ctx) : undefined
      return path ? [path] : []
    }
    case 'core': {
      // A wired Core node contributes its current output - the same hero-take boundary the engine
      // freezes upstream nodes at, so nothing upstream is recomputed.
      const path = item.data.core?.output?.filePath
      return path ? [path] : []
    }
    case 'frame':
    case 'preview': {
      const path = item.frameId ? heroPath(item.frameId, ctx) : undefined
      return path ? [path] : []
    }
    default:
      return []
  }
}

function sourceLabel(item: MoodboardItem, ctx: ThumbContext): string {
  if (item.type === 'controlSpace') return 'Control Space'
  if (item.type === 'loader') return 'Load Assets'
  if (item.type === 'asset' && item.assetId) {
    return ctx.assets.find((a) => a.id === item.assetId)?.name ?? 'Asset'
  }
  if (item.type === 'core') return item.data.core?.type ?? 'Node'
  return 'Frame'
}

function assetPath(assetId: string, ctx: ThumbContext): string | undefined {
  return ctx.assets.find((a) => a.id === assetId)?.filePath
}

/** A frame contributes its chosen take, falling back to the newest when none is pinned. */
function heroPath(frameId: string, ctx: ThumbContext): string | undefined {
  const takes = ctx.takesByFrame[frameId] ?? []
  if (takes.length === 0) return undefined
  const frame = ctx.frames.find((f) => f.id === frameId)
  const hero = frame?.heroTakeId ? takes.find((t) => t.id === frame.heroTakeId) : undefined
  return (hero ?? takes[takes.length - 1]).filePath
}

function isPresent(value: string | undefined): value is string {
  return value != null
}
