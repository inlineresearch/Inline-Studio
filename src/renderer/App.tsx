import { useEffect } from 'react'
import { useProjectStore } from './store/projectStore'
import { useUpdateStore } from './store/updateStore'
import { subscribeToLibraryChanges } from './store/assetStore'
import { ProjectLauncher } from './views/ProjectLauncher/ProjectLauncher'
import { Workspace } from './views/Workspace/Workspace'
import { UpdateBanner } from './components/UpdateBanner'

export function App(): React.JSX.Element {
  const current = useProjectStore((s) => s.current)
  const restoring = useProjectStore((s) => s.restoring)
  const restore = useProjectStore((s) => s.restore)
  const loadRecents = useProjectStore((s) => s.loadRecents)
  const subscribeToUpdates = useUpdateStore((s) => s.subscribeToEvents)

  useEffect(() => {
    void restore()
  }, [restore])

  useEffect(() => {
    void loadRecents()
  }, [loadRecents])

  useEffect(() => subscribeToUpdates(), [subscribeToUpdates])

  useEffect(() => subscribeToLibraryChanges(), [])

  // Hold the first paint until Core has answered, so a restored project does not flash the launcher.
  if (restoring) return <div className="h-screen w-screen bg-panel" />

  return (
    <>
      <UpdateBanner />
      {current ? <Workspace project={current} /> : <ProjectLauncher />}
    </>
  )
}
