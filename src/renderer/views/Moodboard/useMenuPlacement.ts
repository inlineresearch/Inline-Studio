import { useLayoutEffect, useRef, useState } from 'react'

export interface MenuPlacement<T extends HTMLElement> {
  ref: React.RefObject<T | null>
  /** Container-relative position for the popup. */
  style: React.CSSProperties
  /** Cap for the scroll area so the list never runs past the canvas edge. */
  maxHeight: number | undefined
  /** True when the menu was flipped to grow upward from the anchor. */
  flipped: boolean
}

/**
 * Keep a popup anchored to a canvas point fully on screen.
 *
 * Two problems this solves, both only visible near the bottom edge. A menu anchored at the click
 * point runs off the canvas, and the part you cannot see is the part you wanted. Worse, the obvious
 * recovery of scrolling does nothing useful: React Flow owns the wheel, so the canvas zooms instead
 * of the list scrolling. The scroll container needs React Flow's `nowheel` class as well as this.
 *
 * So: flip above the anchor when there is more room up there, and cap the height to whatever room
 * remains either way. Measured after layout rather than guessed, because the list's height depends
 * on how many nodes are installed.
 */
export function useMenuPlacement<T extends HTMLElement>(
  x: number,
  y: number,
  /** True when the caller already grows the menu upward (the toolbar's + button). */
  above: boolean,
): MenuPlacement<T> {
  const ref = useRef<T>(null)
  const [maxHeight, setMaxHeight] = useState<number>()
  const [flipped, setFlipped] = useState(false)

  useLayoutEffect(() => {
    const el = ref.current
    const container = el?.offsetParent as HTMLElement | null
    if (!el || !container) return
    const margin = 12
    const below = container.clientHeight - y - margin
    const above_ = y - margin
    // Flip only when it helps: a menu that fits below stays below, and one that fits nowhere still
    // opens downward rather than off the top.
    const shouldFlip = !above && el.scrollHeight > below && above_ > below
    setFlipped(shouldFlip)
    setMaxHeight(Math.max(160, shouldFlip || above ? above_ : below))
  }, [x, y, above])

  return { ref, style: { left: x, top: y }, maxHeight, flipped }
}
