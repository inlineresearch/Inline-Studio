import type { CoreParamField } from '@shared/coreNodes'

/**
 * One editable widget for an Inline Core node param, driven by the descriptor's field schema. Mirrors
 * the fal `ParamWidget` look (this is the sidebar variant), but typed for `CoreParamField` — it also
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
    const num = Number(value ?? field.default)
    return (
      <label className="flex flex-col gap-1">
        <span className={labelCls}>{field.label}</span>
        <input
          type="number"
          value={num}
          min={field.min}
          max={field.max}
          step={field.step}
          onChange={(e) => onChange(Number(e.target.value))}
          onBlur={() => onCommit(num)}
          className={inputCls}
        />
      </label>
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
