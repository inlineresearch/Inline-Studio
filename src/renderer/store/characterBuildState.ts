/** One architecture's in-flight build, from the pre-run phases through to the adapter landing. */
export interface CharacterBuildState {
  phase: 'preparing' | 'captioning' | 'queued' | 'training' | 'done' | 'failed'
  /** Set once the run exists, which is what later training events are matched against. */
  runId?: string
  fraction: number
  step: number
  totalSteps: number
  /** The trainer's own phase text ("caching latents", "loading model (int8)"). */
  status?: string
  error?: string
}
