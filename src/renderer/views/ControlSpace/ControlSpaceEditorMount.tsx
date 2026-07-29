/**
 * Mounts the Control Space 3D editor only while it is open, so its three.js/react-three-fiber bundle
 * is fetched on demand (the first time a user opens Control Space) rather than in the initial SPA load.
 */
import { Suspense, lazy } from 'react'
import { useControlSpaceStore } from '../../store/controlSpaceStore'

const ControlSpaceEditor = lazy(() => import('./ControlSpaceEditor'))

export function ControlSpaceEditorMount(): React.JSX.Element | null {
  const open = useControlSpaceStore((s) => s.editingItemId !== null)
  if (!open) return null
  return (
    <Suspense fallback={null}>
      <ControlSpaceEditor />
    </Suspense>
  )
}
