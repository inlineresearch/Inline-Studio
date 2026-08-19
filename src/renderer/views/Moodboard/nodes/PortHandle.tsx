/**
 * One coloured port dot with a hover chip naming the port.
 *
 * Shared so every node family reads the same: the Core nodes and the training nodes drew their own
 * handles, and the training ones ended up centred, unnamed and a different colour from the port
 * kind they carry.
 */
import { Handle, Position } from '@xyflow/react'
import { portKindColor, type PortKind } from '@shared/coreNodes'
export function PortHandle({
  id,
  label,
  kind,
  side,
  style,
}: {
  id: string
  label: string
  kind: PortKind
  side: 'input' | 'output'
  style?: React.CSSProperties
}): React.JSX.Element {
  const input = side === 'input'
  return (
    <Handle
      type={input ? 'target' : 'source'}
      id={id}
      position={input ? Position.Left : Position.Right}
      style={{ ...style, background: portKindColor(kind) }}
      className="group !h-3 !w-3 !border-2 !border-surface"
    >
      <span
        className={`pointer-events-none absolute top-1/2 z-50 hidden -translate-y-1/2 whitespace-nowrap rounded bg-black/90 px-1.5 py-0.5 text-[10px] leading-none text-zinc-100 shadow group-hover:block ${
          input ? 'right-full mr-2' : 'left-full ml-2'
        }`}
      >
        {label} <span className="text-zinc-400">· {kind}</span>
      </span>
    </Handle>
  )
}
