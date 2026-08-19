import { useEffect, useRef, useState } from 'react'
import type { CoreParamField } from '@shared/coreNodes'
import { useModelsTreeStore } from '../../store/modelsTreeStore'
import { RefreshIcon } from '../../components/icons'
import { basename, optionsWithPick } from './missingInputs'
import { useAutoCommit } from '../../lib/useAutoCommit'
import type { WiredParam } from './wiredParams'

/** Re-scan the models folders so a file added since Core started reaches this picker. */
function ModelRefreshButton(): React.JSX.Element {
  const refresh = useModelsTreeStore((s) => s.refresh)
  const loading = useModelsTreeStore((s) => s.loading)
  return (
    <button
      type="button"
      onClick={() => void refresh()}
      disabled={loading}
      title="Rescan models folder"
      aria-label="Rescan models folder"
      className="shrink-0 rounded p-1 text-zinc-500 transition-colors hover:bg-black/30 hover:text-zinc-200 disabled:opacity-40"
    >
      <RefreshIcon className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
    </button>
  )
}

/**
 * One editable widget for an Inline Core node param, driven by the descriptor's field schema. Mirrors
 * the fal `ParamWidget` look (this is the sidebar variant), but typed for `CoreParamField` - it also
 * handles the `seed` widget and select options fed from Core's model catalog. `onChange` updates the
 * value while typing; `onCommit` persists it. Text/number fields commit on blur; select/checkbox
 * commit immediately.
 */
export function CoreParamWidget({
  field,
  value,
  wired,
  onChange,
  onCommit,
}: {
  field: CoreParamField
  value: unknown
  /** Set when a wire drives this param; the field then shows what the run will use. */
  wired?: WiredParam
  onChange: (v: string | number | boolean) => void
  onCommit: (v: string | number | boolean) => void
}): React.JSX.Element {
  const labelCls = 'text-[10px] font-medium uppercase tracking-wide text-zinc-500'
  const inputCls =
    'w-full rounded-md border border-border bg-panel px-2 py-1.5 text-xs text-zinc-100 outline-none transition-colors focus:border-accent'
  // Declared above the widget branches: a hook cannot sit behind an early return.
  const { schedule, flush } = useAutoCommit(() => onCommit(String(value ?? field.default ?? '')))

  // Editing it would change nothing, so it reads as what it is: driven from the canvas.
  if (wired && !wired.fallsBack) {
    return (
      <label className="flex flex-col gap-1">
        <span className={labelCls}>{field.label}</span>
        <div
          className={`${inputCls} min-h-[30px] whitespace-pre-wrap break-words border-dashed text-zinc-400`}
        >
          {wired.text || `Set by ${wired.from} when the graph runs`}
        </div>
        <span className="text-[10px] text-zinc-500">From the wired {wired.from} node</span>
      </label>
    )
  }

  if (field.widget === 'boolean') {
    return (
      <label className="flex items-center justify-between gap-2 py-0.5">
        <span className={labelCls}>{field.label}</span>
        <input
          type="checkbox"
          checked={Boolean(value ?? field.default)}
          onChange={(e) => onCommit(e.target.checked)}
          className="accent-accent"
        />
      </label>
    )
  }
  if (field.widget === 'select') {
    const options = field.options ?? []
    // A picker whose empty value genuinely means "none" needs an explicit empty option, or the
    // native select shows the FIRST file while the value is really "" - which reads as picked when
    // it isn't (e.g. the ControlNet dropdown looked set but control was actually off).
    //
    // Only when the *default* is empty, though. Core now resolves a concrete file for the model,
    // VAE and text-encoder pickers, and offering "Auto" beside it hides which file actually ran.
    const needsEmpty = !options.some((o) => o.value === '') && field.default === ''
    const emptyLabel =
      field.optionsFrom === 'controlnet'
        ? 'None'
        : field.optionsFrom === 'characters'
          ? 'No character'
          : 'Auto'
    // Two ways a stored value matches no option: "" from before Core resolved these, and a legacy
    // full path. Either way the browser silently renders the FIRST option, so the card claims a
    // different checkpoint than the run will load.
    const stored = value == null || value === '' ? field.default : value
    const selected =
      typeof stored === 'string' && !options.some((o) => o.value === stored)
        ? basename(stored)
        : stored
    // A pick the catalog lacks stays listed, or the select renders blank and the name the graph
    // arrived with is lost - which is the one thing needed to go and fetch the right file.
    const shown = optionsWithPick(options, String(selected ?? ''))
    return (
      <label className="flex flex-col gap-1">
        <span className={labelCls}>{field.label}</span>
        <div className="flex items-center gap-1">
          <select
            value={String(selected)}
            onChange={(e) => onCommit(e.target.value)}
            className={`${inputCls} min-w-0 flex-1`}
          >
            {needsEmpty && <option value="">{emptyLabel}</option>}
            {shown.map((o) => (
              <option key={o.value} value={o.value}>
                {/* The value has to stay the filename Core resolves by; only the label is
                    friendly. */}
                {field.optionsFrom === 'characters' ? o.label.replace(/\.char$/i, '') : o.label}
              </option>
            ))}
          </select>
          {/* A file added to models/ after Core started is not in this list until a rescan, and
              this is where the user notices it is missing. */}
          {field.optionsFrom && <ModelRefreshButton />}
        </div>
      </label>
    )
  }
  if (field.widget === 'number' || field.widget === 'seed') {
    return (
      <NumberField
        field={field}
        value={value}
        onCommit={onCommit}
        inputCls={inputCls}
        labelCls={labelCls}
      />
    )
  }
  // text + textarea: edit locally, persist on blur.
  const text = String(value ?? field.default ?? '')
  const common = {
    value: text,
    onChange: (e: React.ChangeEvent<HTMLTextAreaElement | HTMLInputElement>) => {
      onChange(e.target.value)
      schedule()
    },
    onBlur: flush,
    placeholder: field.label,
    className: inputCls,
  }
  return (
    <label className="flex flex-col gap-1">
      <span className={labelCls}>{field.label}</span>
      {field.widget === 'textarea' ? (
        <textarea {...common} rows={3} className={`${inputCls} resize-none leading-snug`} />
      ) : (
        <input type="text" {...common} />
      )}
    </label>
  )
}

/**
 * Number/seed input that keeps a free-form text *draft* while the user types, so the field can be
 * cleared entirely and edited mid-value (a bare controlled `Number()` snaps an empty box to 0). The
 * draft only resolves on blur: an empty or unparseable value falls back to the descriptor default,
 * otherwise it commits the parsed number clamped to the field's declared range.
 */
function NumberField({
  field,
  value,
  onCommit,
  inputCls,
  labelCls,
}: {
  field: CoreParamField
  value: unknown
  onCommit: (v: number) => void
  inputCls: string
  labelCls: string
}): React.JSX.Element {
  const external = value ?? field.default
  const [draft, setDraft] = useState<string>(external == null ? '' : String(external))
  // Re-seed the draft when the committed value changes from the outside (e.g. the panel re-seeds for
  // a different node) - but never while the user is editing this same value.
  const lastExternal = useRef(external)
  useEffect(() => {
    if (external !== lastExternal.current) {
      lastExternal.current = external
      setDraft(external == null ? '' : String(external))
    }
  }, [external])

  // Emit while typing so the value registers immediately (not only on blur). A finite entry commits
  // live (unclamped, so mid-typing "5" toward "512" isn't yanked to the min); an empty/partial box is
  // left alone until blur resolves it to the clamped value or the default.
  const emit = (raw: string): void => {
    setDraft(raw)
    if (raw.trim() === '') return
    const parsed = Number(raw)
    if (Number.isFinite(parsed)) {
      lastExternal.current = parsed
      onCommit(parsed)
    }
  }

  const commit = (): void => {
    const parsed = draft.trim() === '' ? Number(field.default) : Number(draft)
    const resolved = Number.isFinite(parsed)
      ? clamp(parsed, field.min, field.max)
      : Number(field.default)
    lastExternal.current = resolved
    setDraft(String(resolved))
    onCommit(resolved)
  }

  return (
    <label className="flex flex-col gap-1">
      <span className={labelCls}>{field.label}</span>
      <input
        type="number"
        value={draft}
        min={field.min}
        max={field.max}
        step={field.step}
        onChange={(e) => emit(e.target.value)}
        onBlur={commit}
        className={inputCls}
      />
    </label>
  )
}

function clamp(n: number, min?: number, max?: number): number {
  if (min != null && n < min) return min
  if (max != null && n > max) return max
  return n
}
