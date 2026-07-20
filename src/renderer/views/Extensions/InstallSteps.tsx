/**
 * The install stepper: every phase the server reports, so the user can see where a slow install
 * is (dependency resolution can take a while) and that it finished.
 */
import type { InstallPhase } from '@shared/extensions'
import { STEPS, stateOf, type StepState } from './installSteps'

function Marker({ state }: { state: StepState }): React.JSX.Element {
  if (state === 'done') {
    return (
      <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-emerald-500/20 text-emerald-300">
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="3"
          className="h-2.5 w-2.5"
        >
          <polyline points="20 6 9 17 4 12" />
        </svg>
      </span>
    )
  }
  if (state === 'failed') {
    return (
      <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-red-500/20 text-red-300">
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="3"
          className="h-2.5 w-2.5"
        >
          <line x1="18" y1="6" x2="6" y2="18" />
          <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      </span>
    )
  }
  if (state === 'active') {
    return (
      <span className="flex h-4 w-4 shrink-0 items-center justify-center">
        <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-emerald-400" />
      </span>
    )
  }
  return (
    <span className="flex h-4 w-4 shrink-0 items-center justify-center">
      <span className="h-2 w-2 rounded-full border border-zinc-600" />
    </span>
  )
}

export function InstallSteps({
  phase,
  seen,
  status,
  failed = false,
}: {
  phase: InstallPhase | 'idle'
  seen: InstallPhase[]
  status: string
  failed?: boolean
}): React.JSX.Element {
  return (
    <ol className="flex flex-col gap-1.5">
      {STEPS.map((step, index) => {
        const state = stateOf(index, phase, seen, failed)
        const tone =
          state === 'done'
            ? 'text-zinc-400'
            : state === 'active'
              ? 'text-zinc-100'
              : state === 'failed'
                ? 'text-red-300'
                : 'text-zinc-600'
        return (
          <li key={step.phase} className={`flex items-center gap-2 text-[12px] ${tone}`}>
            <Marker state={state} />
            <span>{step.label}</span>
            {state === 'active' && status && (
              <span className="truncate text-[11px] text-zinc-500">· {status}</span>
            )}
          </li>
        )
      })}
    </ol>
  )
}
