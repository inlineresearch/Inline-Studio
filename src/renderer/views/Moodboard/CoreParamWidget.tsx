import { useEffect, useRef, useState } from 'react'
import type { CoreParamField } from '@shared/coreNodes'

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
  onChange,
  onCommit,
}: {
  field: CoreParamField
  value: unknown
  onChange: (v: string | number | boolean) => void
  onCommit: (v: string | number | boolean) => void
}): React.JSX.Element {
  const labelCls = 'text-[10px] font-medium uppercase tracking-wide text-zinc-500'
  const inputCls =
    'w-full rounded-md border border-border bg-panel px-2 py-1.5 text-xs text-zinc-100 outline-none transition-colors focus:border-accent'

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
    return (
      <label className="flex flex-col gap-1">
        <span className={labelCls}>{field.label}</span>
        <select
          value={String(value ?? field.default)}
          onChange={(e) => onCommit(e.target.value)}
          className={inputCls}
        >
          {(field.options ?? []).map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
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
    onChange: (e: React.ChangeEvent<HTMLTextAreaElement | HTMLInputElement>) =>
      onChange(e.target.value),
    onBlur: () => onCommit(text),
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
        onChange={(e) => setDraft(e.target.value)}
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
