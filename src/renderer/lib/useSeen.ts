/** Viewport gating for media-heavy lists. */
import { useEffect, useRef, useState } from 'react'

/**
 * True once the element has been near the viewport. Latching matters: a row that scrolls away
 * should keep its decoded frame rather than tear down and refetch on the way back.
 */
export function useSeen<T extends HTMLElement>(): [React.RefObject<T | null>, boolean] {
  const ref = useRef<T>(null)
  const [seen, setSeen] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el || seen) return
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setSeen(true)
          io.disconnect()
        }
      },
      // A screen of lead time, so a scroll finds the frame already there.
      { rootMargin: '600px' },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [seen])
  return [ref, seen]
}
