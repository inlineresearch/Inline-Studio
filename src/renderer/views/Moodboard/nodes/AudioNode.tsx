import { useRef, useState } from 'react'
import type { NodeProps } from '@xyflow/react'
import { NodeFrame } from './NodeFrame'
import { AudioGlyph, NodeBadge, NodeBadgeRow } from './NodeBadge'
import { Waveform } from '../../../components/Waveform'
import type { AssetNodeData } from './nodeData'

export function AudioNode({ id, data, selected }: NodeProps): React.JSX.Element {
  const { src, name, waveform } = data as AssetNodeData
  const audioRef = useRef<HTMLAudioElement>(null)
  const [progress, setProgress] = useState(0)

  const onTimeUpdate = (): void => {
    const el = audioRef.current
    if (el && el.duration > 0) setProgress(el.currentTime / el.duration)
  }
  const seek = (fraction: number): void => {
    const el = audioRef.current
    if (el && el.duration > 0) el.currentTime = fraction * el.duration
  }

  return (
    <>
      <NodeBadgeRow>
        <NodeBadge icon={<AudioGlyph />} title={name}>
          {name}
        </NodeBadge>
      </NodeBadgeRow>
      <NodeFrame id={id} selected={!!selected} subtleSelect>
        <div className="flex h-full w-full flex-col justify-center gap-1 px-2">
          <Waveform
            url={waveform ?? null}
            progress={progress}
            onSeek={seek}
            className="nodrag h-10 w-full text-emerald-400"
          />
          <audio
            ref={audioRef}
            src={src}
            controls
            onTimeUpdate={onTimeUpdate}
            className="nodrag w-full"
          />
        </div>
      </NodeFrame>
    </>
  )
}
