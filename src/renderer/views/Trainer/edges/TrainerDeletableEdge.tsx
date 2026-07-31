import { type EdgeProps } from '@xyflow/react'

import { useTrainerBoardStore } from '../../../store/trainerBoardStore'
import { DeletableEdgeBody } from '../../Moodboard/edges/DeletableEdge'

/** The Studio connector, bound to the Trainer's store: click a link, get a ✕ to unlink it. */
export function TrainerDeletableEdge(props: EdgeProps): React.JSX.Element {
  return <DeletableEdgeBody {...props} disconnect={useTrainerBoardStore((s) => s.disconnect)} />
}
