import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Draft state for a settings sidebar (fal Generate / Inline Core node params).
 *
 * Params edit into a local draft and are persisted explicitly - via the header **Update** button,
 * ⌘/Ctrl+S, or a flush when the panel closes / the open node changes / the panel unmounts. This
 * fixes edits being lost when a click outside closed the panel before the input's blur-commit fired.
 *
 * `persist` is captured at edit time (bound to the node's identity + the just-typed value), so a
 * later flush always writes the correct node's params even after `key` has moved to another node.
 *
 * @param key   Identity of the open node (e.g. `"<id>:<type>"`); changing it reseeds the draft.
 * @param seed  The persisted params to seed the draft from (defaults merged with saved values).
 * @param persist  Writes a full params object to the store.
 */
export function useSettingsDraft(
  key: string | null,
  seed: Record<string, unknown> | undefined,
  persist: (params: Record<string, unknown>) => void,
): {
  local: Record<string, unknown>
  dirty: boolean
  change: (k: string, v: string | number | boolean) => void
  /** Persist pending edits now (Update button / ⌘S). */
  apply: () => void
} {
  const [local, setLocal] = useState<Record<string, unknown>>({})
  const [dirty, setDirty] = useState(false)
  const localRef = useRef(local)
  localRef.current = local

  // A save bound to the node + latest params at edit time; run on any flush.
  const pendingRef = useRef<(() => void) | null>(null)

  const flush = useCallback(() => {
    if (!pendingRef.current) return
    pendingRef.current()
    pendingRef.current = null
    setDirty(false)
  }, [])

  const change = useCallback(
    (k: string, v: string | number | boolean) => {
      const next = { ...localRef.current, [k]: v }
      localRef.current = next
      setLocal(next)
      setDirty(true)
      pendingRef.current = () => persist(next)
    },
    [persist],
  )

  // Reseed when the open node changes - flushing the previous node's pending edits first so they
  // aren't discarded by the reseed.
  useEffect(() => {
    flush()
    setLocal(seed ? { ...seed } : {})
    setDirty(false)
    // Reseed strictly on identity change; `seed` updating in place must not clobber live edits.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  // ⌘/Ctrl+S saves while the panel is open.
  useEffect(() => {
    if (!key) return
    const onKey = (e: KeyboardEvent): void => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') {
        e.preventDefault()
        flush()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [key, flush])

  // Persist any pending edit on unmount (node deleted, tab switch, panel replaced).
  useEffect(() => () => flush(), [flush])

  return { local, dirty, change, apply: flush }
}
