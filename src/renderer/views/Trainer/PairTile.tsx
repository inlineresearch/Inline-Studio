/** One dataset item as a single square: split in half when it is a pair, whole when it is not. */
import type { Asset } from '@shared/types'
import { LazyMedia } from './LazyMedia'

export function PairTile({
  target,
  reference,
  className = '',
}: {
  target?: Asset
  reference?: Asset
  className?: string
}): React.JSX.Element {
  // A pair is one item, so it gets one tile. Asset on the left and its reference on the right,
  // the same order the editor's columns use, so the two surfaces never disagree about which is which.
  return (
    <div className={`relative aspect-square overflow-hidden rounded-sm bg-zinc-900 ${className}`}>
      {reference ? (
        <div className="grid h-full w-full grid-cols-2 gap-px">
          <Half asset={target} compare={reference} />
          <Half asset={reference} compare={target} />
        </div>
      ) : (
        <Half asset={target} />
      )}
    </div>
  )
}

function Half({ asset, compare }: { asset?: Asset; compare?: Asset }): React.JSX.Element {
  if (!asset) return <div className="h-full w-full bg-zinc-900" />
  return <LazyMedia asset={asset} compare={compare} />
}
