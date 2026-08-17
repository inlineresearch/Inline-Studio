import { useCharacterStore } from '../../store/characterStore'

/** A build's live phase. Every stage is named, because a long silent one reads as a hung app. */
export function BuildStatus({ arch }: { arch: string }): React.JSX.Element | null {
  const build = useCharacterStore((s) => s.builds[arch])
  if (!build) return null

  const percent = Math.round((build.fraction ?? 0) * 100)
  const failed = build.phase === 'failed'

  return (
    <div className="mt-2 space-y-1">
      <div className="flex items-center gap-2">
        <span className={`text-[10px] ${failed ? 'text-red-400' : 'text-emerald-400'}`}>
          {label(build)}
        </span>
        {build.totalSteps > 0 && build.phase === 'training' && (
          <span className="ml-auto text-[10px] text-muted">
            {build.step}/{build.totalSteps}
          </span>
        )}
      </div>
      {!failed && (
        <div className="h-0.5 w-full overflow-hidden rounded-full bg-surface">
          <div
            className="h-full bg-emerald-500 transition-[width] duration-300"
            style={{ width: `${percent}%` }}
          />
        </div>
      )}
      {build.error && <p className="text-[10px] text-red-400">{build.error}</p>}
    </div>
  )
}

function label(build: {
  phase: string
  status?: string
  step: number
  totalSteps: number
}): string {
  switch (build.phase) {
    case 'preparing':
      return 'Preparing the training set…'
    case 'captioning':
      return 'Captioning references…'
    case 'queued':
      return 'Queued, waiting for the GPU'
    case 'training':
      // The trainer names its own phases ("caching latents", "loading model (int8)"), so its text
      // wins over a generic one whenever it has sent any.
      return build.status ? capitalise(build.status) : 'Training…'
    case 'done':
      return 'Adapter written into the character'
    case 'failed':
      return 'Training failed'
    default:
      return 'Working…'
  }
}

function capitalise(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1)
}
