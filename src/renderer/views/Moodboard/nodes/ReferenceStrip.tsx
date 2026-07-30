import { resolveMedia } from '../../../lib/media'
import type { CoreInputThumb } from './coreInputThumbs'

/**
 * The numbered reference images wired into a node's list input, docked at the top of its preview.
 *
 * Numbering is the point, not decoration: FLUX.2 resolves "the jacket from image 2" against the
 * order the references arrive in, so the card has to show which is which for the prompt to be
 * writable at all. Order follows the wires, so re-wiring is how you re-order.
 */
export function ReferenceStrip({
  references,
}: {
  references: CoreInputThumb[]
}): React.JSX.Element | null {
  if (references.length === 0) return null
  return (
    <div className="nodrag nowheel absolute inset-x-0 top-0 z-10 flex gap-1 overflow-x-auto bg-gradient-to-b from-black/80 via-black/40 to-transparent px-1.5 pb-5 pt-1.5">
      {references.map((ref) => (
        <div
          key={`${ref.sourceId}-${ref.index}`}
          title={`Reference ${ref.index} (${ref.label}) - call it "image ${ref.index}" in the prompt`}
          className="relative h-9 w-9 shrink-0 overflow-hidden rounded border border-border"
        >
          <img src={resolveMedia(ref.filePath)} alt="" className="h-full w-full object-cover" />
          <span className="absolute bottom-0 right-0 rounded-tl bg-black/75 px-1 text-[9px] font-medium leading-tight text-zinc-200">
            {ref.index}
          </span>
        </div>
      ))}
    </div>
  )
}
