import { useEffect, useRef, useState } from 'react'
import type { TextItemData } from '@shared/types'

/** Preset swatches for quick recolouring. First entry is the light-grey default. */
const PRESET_COLORS = [
  '#e4e4e7', // light grey (default)
  '#ffffff', // white
  '#fbbf24', // amber
  '#f87171', // red
  '#34d399', // green
  '#60a5fa', // blue
  '#c084fc', // purple
  '#18181b', // near-black
]

const MIN_SIZE = 8
const MAX_SIZE = 96

/**
 * Floating formatting bar shown above a selected text node: recolour from a preset
 * palette, bump font size, toggle bold/italic/underline, set alignment, and attach
 * an http(s) link. Marked `nodrag`/`nowheel` so interacting with it never pans the
 * canvas. Every change is pushed up via `onChange` (which persists through the store).
 */
export function TextToolbar({
  text,
  onChange,
}: {
  text: TextItemData
  onChange: (patch: Partial<TextItemData>) => void
}): React.JSX.Element {
  const setSize = (delta: number): void =>
    onChange({ fontSize: Math.min(MAX_SIZE, Math.max(MIN_SIZE, text.fontSize + delta)) })

  const editLink = (): void => {
    const next = window.prompt('Link URL (http/https) - leave blank to remove:', text.link ?? '')
    if (next === null) return // cancelled
    const url = next.trim()
    if (url === '') return onChange({ link: undefined })
    if (!/^https?:\/\//i.test(url)) {
      window.alert('Link must start with http:// or https://')
      return
    }
    onChange({ link: url })
  }

  return (
    <div
      className="nodrag nowheel absolute -top-11 left-1/2 z-20 flex -translate-x-1/2 items-center gap-1 rounded-md border border-border bg-panel px-1.5 py-1 shadow-lg"
      onDoubleClick={(e) => e.stopPropagation()}
    >
      {/* Colour picker (dropdown) */}
      <ColorDropdown color={text.color} onPick={(c) => onChange({ color: c })} />

      <Divider />

      {/* Font size */}
      <button onClick={() => setSize(-2)} title="Smaller" className={btn}>
        A<span className="text-[9px]">−</span>
      </button>
      <span className="w-6 text-center text-[11px] tabular-nums text-zinc-400">
        {text.fontSize}
      </span>
      <button onClick={() => setSize(2)} title="Larger" className={btn}>
        A<span className="text-[9px]">+</span>
      </button>

      <Divider />

      {/* Style toggles */}
      <Toggle active={text.bold} onClick={() => onChange({ bold: !text.bold })} title="Bold">
        <span className="font-bold">B</span>
      </Toggle>
      <Toggle
        active={text.italic}
        onClick={() => onChange({ italic: !text.italic })}
        title="Italic"
      >
        <span className="italic">I</span>
      </Toggle>
      <Toggle
        active={text.underline}
        onClick={() => onChange({ underline: !text.underline })}
        title="Underline"
      >
        <span className="underline">U</span>
      </Toggle>

      <Divider />

      {/* Alignment */}
      {(['left', 'center', 'right'] as const).map((a) => (
        <Toggle key={a} active={text.align === a} onClick={() => onChange({ align: a })} title={a}>
          <AlignGlyph align={a} />
        </Toggle>
      ))}

      <Divider />

      {/* Link */}
      <Toggle
        active={!!text.link}
        onClick={editLink}
        title={text.link ? `Link: ${text.link}` : 'Add link'}
      >
        <LinkGlyph />
      </Toggle>
    </div>
  )
}

/** Swatch button that opens a popover of preset colours; closes on pick or outside click. */
function ColorDropdown({
  color,
  onPick,
}: {
  color: string
  onPick: (c: string) => void
}): React.JSX.Element {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent): void => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        title="Text colour"
        aria-label="Text colour"
        aria-expanded={open}
        className="flex h-6 items-center gap-1 rounded px-1 text-zinc-300 hover:bg-surface"
      >
        <span
          className="h-4 w-4 rounded-full border border-zinc-600"
          style={{ backgroundColor: color }}
        />
        <span className="text-[8px] leading-none">▾</span>
      </button>
      {open && (
        <div className="absolute left-0 top-7 z-30 grid grid-cols-4 gap-1 rounded-md border border-border bg-panel p-1.5 shadow-lg">
          {PRESET_COLORS.map((c) => (
            <button
              key={c}
              onClick={() => {
                onPick(c)
                setOpen(false)
              }}
              title={c}
              aria-label={`Colour ${c}`}
              className={`h-5 w-5 rounded-full border ${
                color.toLowerCase() === c.toLowerCase()
                  ? 'border-white ring-1 ring-accent'
                  : 'border-zinc-600'
              }`}
              style={{ backgroundColor: c }}
            />
          ))}
        </div>
      )}
    </div>
  )
}

const btn =
  'flex h-6 min-w-[1.5rem] items-center justify-center rounded px-1 text-[11px] text-zinc-300 hover:bg-surface'

function Toggle({
  active,
  onClick,
  title,
  children,
}: {
  active: boolean
  onClick: () => void
  title: string
  children: React.ReactNode
}): React.JSX.Element {
  return (
    <button
      onClick={onClick}
      title={title}
      aria-label={title}
      aria-pressed={active}
      className={`flex h-6 w-6 items-center justify-center rounded text-[12px] ${
        active ? 'bg-accent text-panel' : 'text-zinc-300 hover:bg-surface'
      }`}
    >
      {children}
    </button>
  )
}

function Divider(): React.JSX.Element {
  return <span className="mx-0.5 h-5 w-px bg-border" />
}

function AlignGlyph({ align }: { align: 'left' | 'center' | 'right' }): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      className="h-3.5 w-3.5"
    >
      <line x1="3" y1="6" x2="21" y2="6" />
      <line x1={align === 'right' ? 9 : 3} y1="12" x2={align === 'left' ? 15 : 21} y2="12" />
      <line x1="3" y1="18" x2="21" y2="18" />
    </svg>
  )
}

function LinkGlyph(): React.JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className="h-3.5 w-3.5">
      <path d="M3.9 12c0-1.71 1.39-3.1 3.1-3.1h4V7H7c-2.76 0-5 2.24-5 5s2.24 5 5 5h4v-1.9H7c-1.71 0-3.1-1.39-3.1-3.1zM8 13h8v-2H8v2zm9-6h-4v1.9h4c1.71 0 3.1 1.39 3.1 3.1s-1.39 3.1-3.1 3.1h-4V17h4c2.76 0 5-2.24 5-5s-2.24-5-5-5z" />
    </svg>
  )
}
