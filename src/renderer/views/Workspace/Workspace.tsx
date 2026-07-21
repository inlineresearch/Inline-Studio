import type { Project } from '@shared/types'
import { Logo } from '../../components/Logo'
import { SettingsIcon } from '../../components/icons'
import { useProjectStore } from '../../store/projectStore'
import { useAssetStore } from '../../store/assetStore'
import { useMoodboardStore } from '../../store/moodboardStore'
import { useFrameStore } from '../../store/frameStore'
import { useUiStore } from '../../store/uiStore'
import { MoodboardPanel } from '../Moodboard/MoodboardPanel'
import { SettingsPanel } from '../Settings/SettingsPanel'
import { ExtensionsDialog } from '../Extensions/ExtensionsDialog'
import { ContextMenu } from '../../components/ContextMenu'
import { MediaLightbox } from '../../components/MediaLightbox'

/** The main shell: the node canvas plus the Settings drawer. */
export function Workspace({ project }: { project: Project }): React.JSX.Element {
  const settingsOpen = useUiStore((s) => s.settingsOpen)
  const setSettingsOpen = useUiStore((s) => s.setSettingsOpen)
  const closeProject = useProjectStore((s) => s.closeProject)
  const resetAssets = useAssetStore((s) => s.reset)
  const resetBoard = useMoodboardStore((s) => s.reset)
  const resetFrames = useFrameStore((s) => s.reset)

  const onClose = (): void => {
    resetAssets()
    resetBoard()
    resetFrames()
    closeProject()
  }

  return (
    <div className="flex h-full flex-col">
      <header className="relative flex h-12 shrink-0 items-center justify-between border-b border-border bg-surface px-3">
        <div className="flex items-center gap-2.5">
          <button
            onClick={onClose}
            title="Back to your projects"
            className="-m-1 flex items-center gap-2.5 rounded p-1 transition-opacity hover:opacity-75"
          >
            <Logo size={26} />
            <span className="text-sm font-semibold text-white">Inline Studio</span>
          </button>
          <span className="text-zinc-600">/</span>
          <span className="text-sm text-zinc-300">{project.name}</span>
        </div>

        <div className="flex items-center">
          <button
            onClick={() => setSettingsOpen(!settingsOpen)}
            title="Settings"
            aria-label="Settings"
            aria-pressed={settingsOpen}
            className={`flex h-8 w-8 items-center justify-center rounded-md transition-colors ${
              settingsOpen
                ? 'bg-panel text-white'
                : 'text-zinc-400 hover:bg-panel hover:text-zinc-200'
            }`}
          >
            <SettingsIcon className="h-5 w-5" />
          </button>
        </div>
      </header>

      <main className="flex min-h-0 flex-1">
        <div className="relative min-h-0 flex-1">
          <MoodboardPanel />
        </div>

        {settingsOpen && (
          <div className="min-h-0 w-80 shrink-0">
            <SettingsPanel onClose={() => setSettingsOpen(false)} />
          </div>
        )}
      </main>

      <ContextMenu />
      <MediaLightbox />
      <ExtensionsDialog />
    </div>
  )
}
