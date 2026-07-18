import { getNodeDef, modelOwnerLabel } from '@shared/nodes/registry'
import { defaultParams, formatPrice, type PortKind } from '@shared/nodes/types'
import { useGenerationStore } from '../../store/generationStore'
import { AudioGlyph, ImageGlyph, VideoGlyph, XIcon } from './nodes/NodeBadge'

/** Human label for each port kind. */
const KIND_LABEL: Record<PortKind, string> = {
  image: 'Image',
  'image[]': 'Images',
  video: 'Video',
  audio: 'Audio',
  text: 'Text',
  path: 'File',
}

function KindIcon({ kind }: { kind: PortKind }): React.JSX.Element {
  if (kind === 'video') return <VideoGlyph className="h-4 w-4" />
  if (kind === 'audio') return <AudioGlyph className="h-4 w-4" />
  if (kind === 'text') return <span className="text-sm font-bold leading-none">T</span>
  return <ImageGlyph className="h-4 w-4" />
}

/** One input/output port row: kind icon + label + kind name, with an optional required/optional tag. */
function PortRow({
  label,
  kind,
  required,
}: {
  label: string
  kind: PortKind
  required?: boolean
}): React.JSX.Element {
  return (
    <div className="flex items-center gap-2 rounded-md border border-border bg-surface/60 px-2.5 py-2">
      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded text-zinc-300">
        <KindIcon kind={kind} />
      </span>
      <span className="flex min-w-0 flex-1 flex-col">
        <span className="truncate text-[11px] font-medium text-zinc-100">{label}</span>
        <span className="text-[10px] text-zinc-500">{KIND_LABEL[kind]}</span>
      </span>
      {required !== undefined && (
        <span
          className={`shrink-0 rounded-full px-1.5 py-0.5 text-[9px] font-medium ${
            required ? 'bg-amber-500/15 text-amber-300' : 'bg-zinc-500/15 text-zinc-400'
          }`}
        >
          {required ? 'Required' : 'Optional'}
        </span>
      )}
    </div>
  )
}

function Section({
  title,
  children,
}: {
  title: string
  children: React.ReactNode
}): React.JSX.Element {
  return (
    <section className="flex flex-col gap-1.5">
      <h4 className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500">{title}</h4>
      {children}
    </section>
  )
}

/**
 * Right-hand sidebar describing a generation model: what it accepts (inputs), what it produces
 * (outputs), and its estimated cost. Opened from a model row's hint icon in the ModelPicker.
 * Shares the right gutter with the settings panel (opening one closes the other).
 */
export function ModelInfoPanel(): React.JSX.Element | null {
  const modelId = useGenerationStore((s) => s.infoModelId)
  const close = useGenerationStore((s) => s.closeModelInfo)
  const def = modelId ? getNodeDef(modelId) : undefined

  if (!modelId || !def) return null

  const price = def.estimatePrice?.(defaultParams(def)) ?? null

  return (
    <div className="absolute right-0 top-0 z-40 flex h-full w-72 flex-col border-l border-border bg-panel shadow-2xl">
      <header className="flex items-center justify-between gap-2 border-b border-border px-3 py-2">
        <div className="flex min-w-0 flex-col">
          <span className="truncate text-xs font-semibold text-zinc-100">{def.title}</span>
          <span className="truncate text-[10px] text-zinc-500">
            {modelOwnerLabel(def.id)} · {def.category}
          </span>
        </div>
        <button
          onClick={close}
          aria-label="Close model info"
          className="flex h-6 w-6 shrink-0 items-center justify-center rounded text-zinc-400 hover:bg-surface hover:text-zinc-100"
        >
          <XIcon className="h-3.5 w-3.5" />
        </button>
      </header>

      <div className="flex flex-1 flex-col gap-4 overflow-y-auto p-3">
        <Section title="Inputs">
          {/* Every gen node takes a text prompt - a universal input, not listed in def.inputs.
              Prompt-optional models can run without one (the model derives its own). */}
          <PortRow label="Prompt" kind="text" required={!def.promptOptional} />
          {def.inputs.map((p) => (
            <PortRow key={p.id} label={p.label} kind={p.kind} required={p.required} />
          ))}
        </Section>

        <Section title="Outputs">
          {def.outputs.map((o) => (
            <PortRow key={o.id} label={o.label} kind={o.kind} />
          ))}
        </Section>

        <Section title="Cost">
          <div className="flex items-baseline gap-2 rounded-md border border-border bg-surface/60 px-2.5 py-2">
            <span className="text-base font-semibold text-emerald-300">
              {price ? formatPrice(price) : '—'}
            </span>
            <span className="text-[10px] text-zinc-500">
              {price ? 'per generation (estimate)' : 'varies by output'}
            </span>
          </div>
          <p className="text-[10px] leading-relaxed text-zinc-600">
            fal pricing; actual cost may vary with output size and options.
          </p>
        </Section>
      </div>
    </div>
  )
}
