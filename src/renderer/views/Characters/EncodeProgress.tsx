import { useCharacterStore } from '../../store/characterStore'

/** What a running encode is doing; the phase text matters more than the bar. */
export function EncodeProgress({ label }: { label: string }): React.JSX.Element {
  const progress = useCharacterStore((s) => s.progress)
  const percent = Math.round((progress?.fraction ?? 0) * 100)

  return (
    <div className="flex flex-col gap-1 px-2 py-1">
      <span className="text-[11px] text-muted">{progress?.status ?? label}</span>
      <div className="h-0.5 w-full overflow-hidden rounded-full bg-surface">
        <div
          className="h-full bg-emerald-500 transition-[width] duration-300"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  )
}
