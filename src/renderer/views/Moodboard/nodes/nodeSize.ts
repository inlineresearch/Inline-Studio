/**
 * Where a Core node's port dots sit, and the shortest the node can be with all of them inside it.
 *
 * Compact nodes carry no stored height - they hug their content - while the dots are positioned
 * absolutely from the top and bottom edges. A node with more ports than body therefore grew its
 * dots straight out through its own boundary. The layout and this floor share the constants below
 * so the two cannot drift; `nodeSize.test.ts` checks the floor against every served descriptor.
 */
import { isModelPort, type NodeDescriptor } from '@shared/coreNodes'

/** px from the packed edge to the centre of the first dot. */
export const HANDLE_BASE = 18
/** px between the centres of stacked dots. */
export const HANDLE_GAP = 22
/** The dot is `!h-3` (12px) and React Flow centres it on its offset, so it reaches this far. */
export const HALF_DOT = 6

/** How far a stack of `count` dots reaches from the edge it is packed against. */
export function stackExtent(count: number): number {
  return count > 0 ? HANDLE_BASE + (count - 1) * HANDLE_GAP + HALF_DOT : 0
}

/** Vertical padding of the compact body (`py-1.5`), plus the gap between its rows (`gap-1`). */
const BODY_PAD = 12
const ROW_GAP = 4
/** A select or text row: `py-1` on a `text-[10px]` control, plus its border. */
const CONTROL_ROW = 26
/** The bare title line a node with no controls shows instead. */
const LABEL_ROW = 18
/** The Adjust button row (`h-6`), present only when params live behind it. */
const ADJUST_ROW = 24

/** The tallest stack on either side, top-packed and bottom-packed counted separately. */
export function portExtents(descriptor: NodeDescriptor): { top: number; bottom: number } {
  const content = (ports: NodeDescriptor['inputs']): number =>
    ports.filter((p) => !isModelPort(p.kind)).length
  const model = (ports: NodeDescriptor['inputs']): number =>
    ports.filter((p) => isModelPort(p.kind)).length
  return {
    top: stackExtent(Math.max(content(descriptor.inputs), content(descriptor.outputs))),
    bottom: stackExtent(Math.max(model(descriptor.inputs), model(descriptor.outputs))),
  }
}

/**
 * The floor for a compact node: whichever is taller, the room its dots need or the room its body
 * needs. Sized from what the body actually renders - a row per dropdown, otherwise a title line,
 * plus the Adjust row when some params sit behind it.
 */
export function compactNodeMinHeight(descriptor: NodeDescriptor): number {
  const { top, bottom } = portExtents(descriptor)

  // Mirrors GraphNode's compact face: a select shows there by default, anything else opts in.
  const onFace = (p: NodeDescriptor['params'][number]): boolean => p.onFace ?? p.widget === 'select'
  const rows = descriptor.params.filter(onFace).length
  const behindAdjust = descriptor.params.some((p) => !onFace(p))
  const body =
    BODY_PAD +
    (rows > 0 ? rows * CONTROL_ROW + (rows - 1) * ROW_GAP : LABEL_ROW) +
    (behindAdjust ? ADJUST_ROW + ROW_GAP : 0)

  return Math.ceil(Math.max(top + bottom, body))
}
