import { useState } from 'react'
import type { CharacterBuild } from '@shared/types'
import { BUILD_STEP_CHOICES, STEP_NOTES, formatEstimate } from '@shared/characterTraining'
import { useCharacterStore } from '../../store/characterStore'
import { BuildStatus } from './BuildStatus'

/** Which models this character applies to, and training the adapter for the ones that need one. */
export function CharacterBuilds({
  file,
  builds,
  description,
  needsRebuild,
}: {
  file: string
  builds: CharacterBuild[]
  description: string
  needsRebuild: boolean
}): React.JSX.Element | null {
  const [tab, setTab] = useState(builds[0]?.arch ?? '')
  if (builds.length === 0) return null
  const active = builds.find((b) => b.arch === tab) ?? builds[0]

  return (
    <div className="space-y-2">
      <div>
        <span className="text-[10px] uppercase tracking-wide text-muted">Models</span>
        <p className="text-[10px] text-muted">
          One .char can be used across different models. Two ways to build a consistent character.
        </p>
      </div>

      <div className="flex gap-1 border-b border-border">
        {builds.map((row) => (
          <button
            key={row.arch}
            type="button"
            onClick={() => setTab(row.arch)}
            className={`-mb-px border-b-2 px-3 py-1.5 text-[11px] ${
              row.arch === active.arch
                ? 'border-accent text-fg'
                : 'border-transparent text-muted hover:text-fg'
            }`}
          >
            {row.label}
          </button>
        ))}
      </div>

      <div className="grid gap-2 md:grid-cols-2">
        {active.reference && (
          <ReferenceMode file={file} build={active} needsRebuild={needsRebuild} />
        )}
        <LoraMode file={file} build={active} description={description} />
      </div>
    </div>
  )
}

/** Present tense on purpose: the reference tier needs no build, which is its whole advantage. */
function ReferenceMode({
  file,
  build,
  needsRebuild,
}: {
  file: string
  build: CharacterBuild
  needsRebuild: boolean
}): React.JSX.Element {
  const rebuild = useCharacterStore((s) => s.rebuild)
  const setApplyMode = useCharacterStore((s) => s.setApplyMode)
  const busy = useCharacterStore((s) => s.busy)
  const stale = needsRebuild && !busy
  const chosen = build.mode === 'reference'

  return (
    <div
      className={`flex flex-col rounded border p-2.5 ${
        chosen ? 'border-emerald-600/40 bg-emerald-500/5' : 'border-border bg-surface/60'
      }`}
    >
      <label className="flex cursor-pointer items-center gap-1.5">
        <input
          type="radio"
          checked={chosen}
          onChange={() => void setApplyMode(file, build.arch, 'reference')}
          className="accent-emerald-500"
        />
        <span className="text-[11px] text-fg">Reference identity</span>
      </label>
      <p className="mt-1.5 text-[10px] text-muted">
        The model reads the references directly. Nothing to train, and it scores higher on faces
        than a trained adapter does.
      </p>

      <div className="mt-auto flex items-center gap-2 pt-2">
        <span className={`text-[10px] ${stale ? 'text-amber-400' : 'text-emerald-400'}`}>
          {busy ? 'Building' : stale ? 'Needs rebuild' : 'Ready'}
        </span>
        {stale && (
          <button
            type="button"
            onClick={() => void rebuild(file)}
            className="ml-auto rounded border border-border px-2 py-0.5 text-[10px] text-muted hover:bg-surface hover:text-fg"
          >
            Rebuild
          </button>
        )}
      </div>
    </div>
  )
}

function LoraMode({
  file,
  build,
  description,
}: {
  file: string
  build: CharacterBuild
  description: string
}): React.JSX.Element {
  const startBuild = useCharacterStore((s) => s.build)
  const cancelBuild = useCharacterStore((s) => s.cancelBuild)
  const setApplyMode = useCharacterStore((s) => s.setApplyMode)
  const active = useCharacterStore((s) => s.builds[build.arch])
  const [open, setOpen] = useState(false)
  const [steps, setSteps] = useState<number>(1200)
  const [autoCaption, setAutoCaption] = useState(false)
  const [starting, setStarting] = useState(false)
  const describable = description.trim().length > 0
  const blocked = !describable || !build.baseReady
  const estimate = formatEstimate(build.arch, steps)
  const running = Boolean(active) && !['done', 'failed'].includes(active?.phase ?? '')

  const start = async (): Promise<void> => {
    setStarting(true)
    await startBuild(file, build.arch, { steps, autoCaption })
    setStarting(false)
  }

  return (
    <div className="flex flex-col rounded border border-border bg-surface/60 p-2.5">
      {build.reference ? (
        <label className="flex cursor-pointer items-center gap-1.5">
          <input
            type="radio"
            checked={build.mode === 'lora'}
            disabled={build.lora !== 'ready'}
            onChange={() => void setApplyMode(file, build.arch, 'lora')}
            className="accent-emerald-500"
          />
          <span className="text-[11px] text-fg">Train character LoRA</span>
        </label>
      ) : (
        <span className="text-[11px] text-fg">Train character LoRA</span>
      )}
      <span className="mt-0.5 text-[10px] text-muted">{state(build)}</span>

      <BuildStatus arch={build.arch} />

      {!open ? (
        <button
          type="button"
          disabled={blocked}
          title={
            !build.baseReady
              ? 'The base checkpoint is not downloaded yet. Add it from a node model popup first.'
              : describable
                ? undefined
                : 'Add a description first: the adapter binds to it, not to a trigger word.'
          }
          onClick={() => setOpen(true)}
          className="mt-2 self-start rounded border border-border px-2 py-1 text-[10px] text-muted hover:bg-surface hover:text-fg disabled:opacity-40"
        >
          {build.lora === 'none' ? 'Train' : 'Retrain'}
        </button>
      ) : (
        <div className="mt-2 space-y-2 border-t border-border pt-2">
          <div className="space-y-0.5">
            <div className="flex items-baseline gap-2">
              <span className="text-[9px] uppercase tracking-wide text-muted">Steps</span>
              <span className="text-[9px] text-muted">{STEP_NOTES[steps]}</span>
              <span className="ml-auto text-[9px] text-muted">{estimate ?? ''}</span>
            </div>
            {/* Indexed, not a raw step range: only these three values have a measured score. */}
            <input
              type="range"
              min={0}
              max={BUILD_STEP_CHOICES.length - 1}
              step={1}
              value={BUILD_STEP_CHOICES.indexOf(steps as (typeof BUILD_STEP_CHOICES)[number])}
              onChange={(e) => setSteps(BUILD_STEP_CHOICES[Number(e.target.value)])}
              className="h-1 w-full accent-emerald-500"
            />
            <div className="flex justify-between text-[9px] text-muted">
              {BUILD_STEP_CHOICES.map((choice) => (
                <span key={choice} className={choice === steps ? 'text-fg' : undefined}>
                  {choice}
                </span>
              ))}
            </div>
          </div>

          <label className="flex cursor-pointer items-center gap-2 text-[10px] text-muted">
            <input
              type="checkbox"
              checked={autoCaption}
              onChange={(e) => setAutoCaption(e.target.checked)}
              className="accent-emerald-500"
            />
            Caption each reference automatically
          </label>

          {running ? (
            <button
              type="button"
              onClick={() => void cancelBuild(build.arch)}
              className="rounded bg-red-600 px-2.5 py-1 text-[10px] text-white hover:bg-red-500"
            >
              Cancel
            </button>
          ) : (
            <button
              type="button"
              disabled={starting}
              onClick={() => void start()}
              className="rounded bg-emerald-600 px-2.5 py-1 text-[10px] text-white hover:bg-emerald-500 disabled:opacity-40"
            >
              {starting ? 'Starting…' : 'Start training'}
            </button>
          )}
        </div>
      )}
    </div>
  )
}

function state(build: CharacterBuild): string {
  if (!build.baseReady) return 'Base checkpoint not downloaded'
  if (build.lora === 'ready') return 'Trained adapter ready'
  if (build.lora === 'stale') return 'Out of date: the references changed since it trained'
  return build.reference ? 'Not trained' : 'Required: this model has no reference channel'
}
