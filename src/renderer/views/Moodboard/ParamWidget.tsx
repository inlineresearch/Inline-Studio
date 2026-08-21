import { useEffect, useRef, useState } from 'react'
import type { ParamField } from '@shared/nodes/types'
import { useAutoCommit } from '../../lib/useAutoCommit'

/**
 * One editable param widget, driven by a NodeDef's declarative field schema.
 * `onChange` updates the value locally (typing); `onCommit` persists it. Text fields commit on
 * blur; selects / numbers / checkboxes commit immediately.
 */
export function ParamWidget({
  field,
  value,
  onChange,
  onCommit,
}: {
  field: ParamField
  value: unknown
  onChange: (v: string | number | boolean) => void
  onCommit: (v: string | number | boolean) => void
}): React.JSX.Element {
  const labelCls = 'text-[10px] font-medium uppercase tracking-wide text-zinc-500'
  const inputCls =
    'w-full rounded-md border border-border bg-panel px-2 py-1.5 text-xs text-zinc-100 outline-none transition-colors focus:border-accent'
  // Declared above the widget branches: a hook cannot sit behind an early return.
  const { schedule, flush } = useAutoCommit(() => onCommit(String(value ?? '')))
  if (field.widget === 'textarea' || field.widget === 'text') {
    const text = String(value ?? '')
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
  if (field.widget === 'select') {
    return (
      <label className="flex flex-col gap-1">
        <span className={labelCls}>{field.label}</span>
        <select
          value={String(value ?? field.default)}
          onChange={(e) => onCommit(e.target.value)}
          className={inputCls}
        >
          {field.options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </label>
    )
  }
  if (field.widget === 'number') {
    return (
      <NumberField
        field={field}
        value={value}
        onChange={onChange}
        onCommit={onCommit}
        inputCls={inputCls}
        labelCls={labelCls}
      />
    )
  }
  return (
    <label className="flex items-center justify-between gap-2 py-0.5">
      <span className={labelCls}>{field.label}</span>
      <input
        type="checkbox"
        checked={Boolean(value)}
        onChange={(e) => onCommit(e.target.checked)}
        className="accent-accent"
      />
    </label>
  )
}

/**
 * Number input that keeps a free-form text *draft* while the user types, so the box can be cleared
 * entirely and edited mid-value (a bare controlled `Number()` snaps an empty box to 0, trapping you
 * into editing around it). The draft resolves on blur: empty or unparseable falls back to the
 * field's default, anything else commits parsed and clamped to the declared range.
 *
 * Mirrors `CoreParamWidget`'s NumberField; the two panels stay separate because their field schemas
 * are different types.
 */
function NumberField({
  field,
  value,
  onChange,
  onCommit,
  inputCls,
  labelCls,
}: {
  field: Extract<ParamField, { widget: 'number' }>
  value: unknown
  onChange: (v: number) => void
  onCommit: (v: number) => void
  inputCls: string
  labelCls: string
}): React.JSX.Element {
  const external = value ?? field.default
  const [draft, setDraft] = useState<string>(external == null ? '' : String(external))
  // Re-seed when the committed value changes from outside (the panel switching to another node),
  // but never while the user is mid-edit of this same value.
  const lastExternal = useRef(external)
  useEffect(() => {
    if (external !== lastExternal.current) {
      lastExternal.current = external
      setDraft(external == null ? '' : String(external))
    }
  }, [external])

  // Emit while typing so an outside click that persists the panel picks up the edit. Unclamped
  // here, so typing "1" on the way to "12" is not yanked up to a minimum of 5.
  const emit = (raw: string): void => {
    setDraft(raw)
    if (raw.trim() === '') return
    const parsed = Number(raw)
    if (Number.isFinite(parsed)) {
      lastExternal.current = parsed
      onChange(parsed)
    }
  }

  const commit = (): void => {
    const parsed = draft.trim() === '' ? Number(field.default) : Number(draft)
    const resolved = Number.isFinite(parsed) ? clamp(parsed, field.min, field.max) : field.default
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
