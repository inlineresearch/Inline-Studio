import type { ParamField } from '@shared/nodes/types'

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
  if (field.widget === 'textarea' || field.widget === 'text') {
    const text = String(value ?? '')
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
      <label className="flex flex-col gap-1">
        <span className={labelCls}>{field.label}</span>
        <input
          type="number"
          value={Number(value ?? field.default)}
          min={field.min}
          max={field.max}
          step={field.step}
          onChange={(e) => onChange(Number(e.target.value))}
          onBlur={() => onCommit(Number(value ?? field.default))}
          className={inputCls}
        />
      </label>
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
