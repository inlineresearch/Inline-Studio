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

/** Dots stack from the top edge, so a node's ports read in a fixed order however tall it grows. */
export function topStyle(index: number): { top: number } {
  return { top: HANDLE_BASE + index * HANDLE_GAP }
}

/** Model-family ports pack from the bottom, so engine wiring reads as one band along it. */
export function bottomStyle(index: number): { top: 'auto'; bottom: number } {
  return { top: 'auto', bottom: HANDLE_BASE + index * HANDLE_GAP }
}

/** How far a stack of `count` dots reaches from the edge it is packed against. */
export function stackExtent(count: number): number {
  return count > 0 ? HANDLE_BASE + (count - 1) * HANDLE_GAP + HALF_DOT : 0
}

/** Vertical padding of the compact body (`py-1.5`), plus the gap between its rows (`gap-1`). */
const BODY_PAD = 12
const ROW_GAP = 4
/** A select or text row: `py-1` on a `text-[10px]` control, plus its border. */
const CONTROL_ROW = 26
/** A field's heading above its control (`text-[9px]` plus its margin). */
const HEADING_ROW = 13
/** A textarea shows four lines, so it cannot be counted as one control row. */
const TEXTAREA_ROW = 62
/** A sweep publishes a table under its params; without this it opens clipped to a few rows. */
const RESULTS_BODY = 150
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
  const faced = descriptor.params.filter(onFace)
  const rows = faced.length
  const controls = faced.reduce(
    (total, p) => total + HEADING_ROW + (p.widget === 'textarea' ? TEXTAREA_ROW : CONTROL_ROW),
    0,
  )
  const behindAdjust = descriptor.params.some((p) => !onFace(p))
  const body =
    BODY_PAD +
    (rows > 0 ? controls + (rows - 1) * ROW_GAP : LABEL_ROW) +
    (behindAdjust ? ADJUST_ROW + ROW_GAP : 0) +
    (publishesResults(descriptor) ? RESULTS_BODY : 0)

  return Math.ceil(Math.max(top + bottom, body))
}

/** Whether a node renders its own findings under its params, which needs room the params do not. */
export function publishesResults(descriptor: NodeDescriptor): boolean {
  return descriptor.type === 'character/finetune'
}

/** How wide a compact node opens. A textarea or a results table is unreadable at the 200 default. */
export function compactNodeWidth(descriptor: NodeDescriptor): number {
  const onFace = (p: NodeDescriptor['params'][number]): boolean => p.onFace ?? p.widget === 'select'
  if (publishesResults(descriptor)) return 420
  if (descriptor.params.filter(onFace).some((p) => p.widget === 'textarea')) return 340
  return 200
}
