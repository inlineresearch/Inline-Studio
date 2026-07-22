/** Phase ordering and per-step state for the install stepper. */
import type { InstallPhase } from '@shared/extensions'

/** The phases the installer actually emits, in order. `install` is folded into `resolve`. */
export const STEPS: { phase: InstallPhase; label: string }[] = [
  { phase: 'fetch', label: 'Download the repository' },
  { phase: 'validate', label: 'Check the manifest' },
  { phase: 'scan', label: 'Review the code' },
  { phase: 'preflight', label: 'Check for conflicts' },
  { phase: 'resolve', label: 'Resolve dependencies' },
  { phase: 'lock', label: 'Record the install' },
  { phase: 'register', label: 'Load nodes' },
  { phase: 'activate', label: 'Activate' },
]

export type StepState = 'done' | 'active' | 'failed' | 'pending'

export function stateOf(
  index: number,
  current: InstallPhase | 'idle',
  seen: InstallPhase[],
  failed: boolean,
): StepState {
  const step = STEPS[index]!.phase
  const currentIndex = STEPS.findIndex((s) => s.phase === current)
  if (failed && step === current) return 'failed'
  // `done` means the whole run finished, so every step before it is complete.
  if (current === 'done') return 'done'
  if (seen.includes(step) && currentIndex > index) return 'done'
  if (step === current) return 'active'
  return currentIndex > index ? 'done' : 'pending'
}
