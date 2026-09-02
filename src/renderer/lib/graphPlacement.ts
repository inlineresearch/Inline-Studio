/**
 * Where an imported graph lands: clear of everything already on the canvas.
 *
 * Rightward rather than downward because pipelines chain left to right, so a new island on the
 * right reads as "the next thing" instead of interrupting a chain.
 */
import type { MoodboardItem } from '@shared/types'

export interface Point {
  x: number
  y: number
}

/** Enough that the two islands never read as one graph at a normal zoom. */
export const IMPORT_GUTTER = 240

/** Recipe items carry their own size; a missing one falls back to the compact node default. */
interface Sized {
  x: number
  y: number
  width?: number
  height?: number
}

const FALLBACK_WIDTH = 200
const FALLBACK_HEIGHT = 120

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

/**
 * The rectangle enclosing these items, from their stored geometry.
 *
 * Deliberately not React Flow's measured sizes: a node that has not rendered yet measures as
 * nothing, so framing an import off measurements races the render and lands on the flow origin.
 */
export function boundsOf(items: readonly Sized[]): Rect | null {
  if (items.length === 0) return null
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const it of items) {
    minX = Math.min(minX, it.x)
    minY = Math.min(minY, it.y)
    maxX = Math.max(maxX, it.x + (it.width ?? FALLBACK_WIDTH))
    maxY = Math.max(maxY, it.y + (it.height ?? FALLBACK_HEIGHT))
  }
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY }
}

/**
 * The drop point for a graph whose items are `incoming`, on a canvas already holding `existing`.
 *
 * The returned point is where the recipe's *target* item lands, which is what
 * `buildGraphFromRecipe` offsets everything else from, so the incoming graph's own layout has to be
 * taken into account: the offset is measured from its top-left corner, not from the target.
 */
export function placeImport(
  existing: readonly MoodboardItem[],
  incoming: readonly Sized[],
  target: Sized,
  centre: Point,
): Point {
  const taken = boundsOf(existing)
  const arriving = boundsOf(incoming)
  // An empty canvas has nothing to sit clear of, so the import lands where the user is looking.
  if (!taken || !arriving) return centre

  return {
    x: taken.x + taken.width + IMPORT_GUTTER + (target.x - arriving.x),
    y: taken.y + (target.y - arriving.y),
  }
}

/** The recipe item everything else is offset from: the named target, else the first item. */
export function recipeTarget<T extends Sized & { id: string }>(
  items: readonly T[],
  targetId: string | undefined,
): T | undefined {
  return items.find((i) => i.id === targetId) ?? items[0]
}
