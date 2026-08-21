/** Mounts the training Adjust sidebar off the same slot the Core one uses, so only one can show. */
import { useGenerationStore } from '../../store/generationStore'
import { useMoodboardStore } from '../../store/moodboardStore'
import { TrainerSettingsPanel } from './TrainerSettingsPanel'

const ADJUSTABLE = new Set(['train/lora', 'train/caption'])

export function TrainingSettingsMount(): React.JSX.Element | null {
  const itemId = useGenerationStore((s) => s.settingsCoreItemId)
  const item = useMoodboardStore((s) => s.items.find((i) => i.id === itemId))
  if (!item || !ADJUSTABLE.has(item.type)) return null
  return <TrainerSettingsPanel itemId={item.id} />
}
