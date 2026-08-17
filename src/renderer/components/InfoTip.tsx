import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { InfoIcon } from './icons'

/** A setting's explanation behind an info button, portalled so a scrolling panel cannot clip it. */
const WIDTH = 256
const MARGIN = 8
const GAP = 6

export function InfoTip({
  label,
  children,
}: {
  /** Names the thing being explained, for the button's accessible label. */
  label: string
  children: React.ReactNode
}): React.JSX.Element {
  const [open, setOpen] = useState(false)
  const [at, setAt] = useState<{ left: number; top: number } | null>(null)
  const button = useRef<HTMLButtonElement>(null)
  const panel = useRef<HTMLDivElement>(null)

  const place = useCallback((): void => {
    const anchor = button.current?.getBoundingClientRect()
    if (!anchor) return
    // Prefer the right of the button; flip left when that would overflow.
    const wantsFlip = anchor.right + GAP + WIDTH > window.innerWidth - MARGIN
    const raw = wantsFlip ? anchor.left - GAP - WIDTH : anchor.right + GAP
    const left = Math.max(MARGIN, Math.min(raw, window.innerWidth - WIDTH - MARGIN))

    const height = panel.current?.offsetHeight ?? 0
    const top = Math.max(MARGIN, Math.min(anchor.top, window.innerHeight - height - MARGIN))
    setAt({ left, top })
  }, [])

  useLayoutEffect(() => {
    if (open) place()
  }, [open, place])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') setOpen(false)
    }
    // Pointer-down, not click: closing on click would fire before a link inside the panel resolves.
    const onAway = (e: PointerEvent): void => {
      const target = e.target as Node
      if (!button.current?.contains(target) && !panel.current?.contains(target)) setOpen(false)
    }
    // Capture, so a scroll in any ancestor repositions rather than leaving the panel behind.
    window.addEventListener('keydown', onKey)
    window.addEventListener('pointerdown', onAway)
    window.addEventListener('scroll', place, true)
    window.addEventListener('resize', place)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('pointerdown', onAway)
      window.removeEventListener('scroll', place, true)
      window.removeEventListener('resize', place)
    }
  }, [open, place])

  return (
    <>
      <button
        ref={button}
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-label={`About ${label}`}
        className={`inline-flex h-4 w-4 items-center justify-center rounded-full transition ${
          open ? 'text-zinc-200' : 'text-zinc-500 hover:text-zinc-300'
        }`}
      >
        <InfoIcon className="h-3.5 w-3.5" />
      </button>

      {open &&
        createPortal(
          <div
            ref={panel}
            role="note"
            style={{ left: at?.left ?? -9999, top: at?.top ?? -9999, width: WIDTH }}
            className="fixed z-50 rounded-md border border-border bg-panel p-2.5 text-[11px] leading-relaxed text-zinc-400 shadow-xl"
          >
            {children}
          </div>,
          document.body,
        )}
    </>
  )
}
