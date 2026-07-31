/**
 * The one-time coaching that follows a starter card: two arrows, at Run graph and at Edit params.
 *
 * Anchoring is by measurement rather than layout, because both targets move with the canvas. The
 * Run pill is the awkward one: it is a React Flow NodeToolbar that only mounts while the node is
 * the selected graph's run target, so it can vanish under us if the board refreshes. When it does,
 * we fall back to the node's own top-right corner rather than draw an arrow pointing at nothing.
 */
import { useEffect, useRef, useState } from 'react'

import { CoachMark, type Point } from '../../../components/CoachMark'
import { useOnboardingStore } from '../../../store/onboardingStore'

/** Measured every frame while visible: two rects, and only during onboarding. */
interface Anchors {
  run: Point | null
  adjust: Point | null
}

const AUTO_DISMISS_MS = 20_000

export function FirstRunHints({
  wrapperRef,
}: {
  wrapperRef: React.RefObject<HTMLDivElement | null>
}): React.JSX.Element | null {
  const target = useOnboardingStore((s) => s.hintTarget)
  const dismiss = useOnboardingStore((s) => s.dismissHints)
  const [anchors, setAnchors] = useState<Anchors>({ run: null, adjust: null })
  const frame = useRef<number>(0)

  // Track both controls while the hints are up. rAF rather than ResizeObserver plus onMove: those
  // miss inertial pan and the animated fitView frames, so the arrow visibly lags right after a click.
  useEffect(() => {
    if (!target) return
    const wrapper = wrapperRef.current
    if (!wrapper) return

    const toLocal = (el: Element | null): Point | null => {
      if (!el) return null
      const r = el.getBoundingClientRect()
      const w = wrapper.getBoundingClientRect()
      return { x: r.left + r.width / 2 - w.left, y: r.top + r.height / 2 - w.top }
    }
    const node = () => wrapper.querySelector(`.react-flow__node[data-id="${target.itemId}"]`)
    const tick = (): void => {
      setAnchors({
        run: toLocal(document.querySelector('[data-run-toolbar]')) ?? nodeCorner(node(), wrapper),
        // Not unique on its own: every node with settings renders one, so scope to this node.
        adjust: toLocal(node()?.querySelector('[data-gen-settings-toggle]') ?? null),
      })
      frame.current = requestAnimationFrame(tick)
    }
    frame.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame.current)
  }, [target, wrapperRef])

  // Any deliberate action means the lesson landed. Escape and a timeout stop it outstaying welcome.
  useEffect(() => {
    if (!target) return
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') dismiss()
    }
    const onDown = (e: PointerEvent): void => {
      const el = e.target as HTMLElement
      if (el.closest('[data-run-toolbar], [data-gen-settings-toggle]')) dismiss()
    }
    const timer = window.setTimeout(dismiss, AUTO_DISMISS_MS)
    window.addEventListener('keydown', onKey)
    window.addEventListener('pointerdown', onDown, true)
    return () => {
      window.clearTimeout(timer)
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('pointerdown', onDown, true)
    }
  }, [target, dismiss])

  if (!target) return null
  return (
    <div className="pointer-events-none absolute inset-0 z-30">
      {anchors.run && (
        <CoachMark
          target={anchors.run}
          step={1}
          title="Run graph"
          body="Renders this graph. The result becomes a take, and running again adds another rather than replacing it."
          side="left"
          onDismiss={dismiss}
        />
      )}
      {anchors.adjust && (
        <CoachMark
          target={anchors.adjust}
          step={2}
          title="Edit params"
          body="Opens size, steps, guidance and the model files in a side panel."
          side="right"
        />
      )}
    </div>
  )
}

/** Fallback anchor: the node's top-right, where the Run pill sits when it is mounted. */
function nodeCorner(node: Element | null | undefined, wrapper: HTMLElement): Point | null {
  if (!node) return null
  const r = node.getBoundingClientRect()
  const w = wrapper.getBoundingClientRect()
  return { x: r.right - w.left, y: r.top - w.top - 10 }
}
