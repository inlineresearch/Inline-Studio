/**
 * The Trainer tab's node canvas: Load Dataset → Caption → Trainer → Graph, plus the utility
 * Resource node. Same React Flow + node-card design as the Studio moodboard, but backed by the
 * `trainer` surface so the two boards stay separate.
 */
import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Background,
  BackgroundVariant,
  ConnectionMode,
  Controls,
  ReactFlow,
  ReactFlowProvider,
  applyNodeChanges,
  useReactFlow,
  type Edge,
  type Node,
  type NodeChange,
  type NodeTypes,
  type OnConnect,
} from '@xyflow/react'
import type { MoodboardItem } from '@shared/types'
import { useTrainerBoardStore, type TrainerNodeKind } from '../../store/trainerBoardStore'
import { BoardActionsContext } from '../Moodboard/nodes/boardActions'
import { ResourceNode } from '../Moodboard/nodes/ResourceNode'
import {
  CaptionGlyph,
  ChartIcon,
  CpuIcon,
  LayersIcon,
  WandIcon,
} from '../Moodboard/nodes/NodeBadge'
import { CaptionNode } from './nodes/CaptionNode'
import { LossGraphNode } from './nodes/LossGraphNode'
import { TrainDatasetNode } from './nodes/TrainDatasetNode'
import { TrainerNode } from './nodes/TrainerNode'

const nodeTypes: NodeTypes = {
  trainDataset: TrainDatasetNode,
  caption: CaptionNode,
  trainer: TrainerNode,
  lossGraph: LossGraphNode,
  resource: ResourceNode,
}

/** The add-node menu, grouped the way the Studio menu groups Core nodes. */
const ADDABLE: Array<{
  category: string
  kind: TrainerNodeKind
  label: string
  icon: React.JSX.Element
}> = [
  { category: 'Training', kind: 'trainDataset', label: 'Load Dataset', icon: <LayersIcon /> },
  { category: 'Training', kind: 'caption', label: 'Caption', icon: <CaptionGlyph /> },
  { category: 'Training', kind: 'trainer', label: 'Train LoRA', icon: <WandIcon /> },
  { category: 'Training', kind: 'lossGraph', label: 'Graph', icon: <ChartIcon /> },
  { category: 'Utility', kind: 'resource', label: 'Resources', icon: <CpuIcon /> },
]

function toNode(item: MoodboardItem): Node {
  return {
    id: item.id,
    type: item.type,
    position: { x: item.x, y: item.y },
    style: { width: item.width, height: item.height },
    data: {},
  }
}

function AddMenu({ onAdd }: { onAdd: (kind: TrainerNodeKind) => void }): React.JSX.Element {
  const [open, setOpen] = useState(false)
  const groups = useMemo(() => {
    const out = new Map<string, typeof ADDABLE>()
    for (const entry of ADDABLE)
      out.set(entry.category, [...(out.get(entry.category) ?? []), entry])
    return [...out.entries()]
  }, [])

  return (
    <div className="absolute left-3 top-3 z-10">
      <button
        onClick={() => setOpen((o) => !o)}
        className="rounded-md border border-border bg-panel/95 px-3 py-1.5 text-xs font-medium text-zinc-200 shadow-sm backdrop-blur hover:bg-panel"
      >
        + Add node
      </button>
      {open && (
        <div className="mt-1 w-52 rounded-md border border-border bg-panel/95 p-1 shadow-lg backdrop-blur">
          {groups.map(([category, entries]) => (
            <div key={category}>
              <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-zinc-500">
                {category}
              </div>
              {entries.map((e) => (
                <button
                  key={e.kind}
                  onClick={() => {
                    onAdd(e.kind)
                    setOpen(false)
                  }}
                  className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-xs text-zinc-200 hover:bg-surface"
                >
                  <span className="text-zinc-400">{e.icon}</span>
                  {e.label}
                </button>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function Canvas(): React.JSX.Element {
  const items = useTrainerBoardStore((s) => s.items)
  const connectors = useTrainerBoardStore((s) => s.connectors)
  const error = useTrainerBoardStore((s) => s.error)
  const load = useTrainerBoardStore((s) => s.load)
  const addNode = useTrainerBoardStore((s) => s.addNode)
  const updateItem = useTrainerBoardStore((s) => s.updateItem)
  const deleteItem = useTrainerBoardStore((s) => s.deleteItem)
  const connect = useTrainerBoardStore((s) => s.connect)
  const disconnect = useTrainerBoardStore((s) => s.disconnect)

  const [nodes, setNodes] = useState<Node[]>([])
  const rf = useReactFlow()

  useEffect(() => {
    void load()
  }, [load])

  /** Drop a new node into the visible area, laid out on a grid so repeated adds never land on top
   * of each other (nodes are ~300px wide, so the step has to clear a whole card). */
  const addAtViewport = (kind: TrainerNodeKind): void => {
    const { x, y, zoom } = rf.getViewport()
    const col = items.length % 3
    const row = Math.floor(items.length / 3) % 3
    void addNode(kind, (-x + 120) / zoom + col * 360, (-y + 120) / zoom + row * 320)
  }

  // Mirror store items into React Flow nodes, preserving in-flight drag positions.
  useEffect(() => {
    setNodes((prev) => {
      const byId = new Map(prev.map((n) => [n.id, n]))
      return items.map((item) => {
        const existing = byId.get(item.id)
        return existing
          ? { ...existing, style: { width: item.width, height: item.height } }
          : toNode(item)
      })
    })
  }, [items])

  const edges: Edge[] = useMemo(
    () =>
      connectors.map((c) => ({
        id: c.id,
        source: c.fromItemId,
        target: c.toItemId,
        sourceHandle: (c.data?.sourceHandle as string | null) ?? undefined,
        targetHandle: (c.data?.targetHandle as string | null) ?? undefined,
      })),
    [connectors],
  )

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => setNodes((nds) => applyNodeChanges(changes, nds)),
    [],
  )

  const onConnect: OnConnect = useCallback(
    (c) => {
      if (!c.source || !c.target) return
      void connect(c.source, c.target, c.sourceHandle ?? null, c.targetHandle ?? null)
    },
    [connect],
  )

  const boardActions = useMemo(() => ({ updateItem, deleteItem }), [updateItem, deleteItem])

  return (
    <div className="relative h-full min-h-0 w-full">
      {error && (
        <div className="absolute left-1/2 top-3 z-20 -translate-x-1/2 rounded-md bg-red-500/10 px-3 py-1.5 text-xs text-red-400">
          {error}
        </div>
      )}
      <AddMenu onAdd={addAtViewport} />
      <BoardActionsContext.Provider value={boardActions}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          connectionMode={ConnectionMode.Loose}
          deleteKeyCode={['Backspace', 'Delete']}
          zoomOnDoubleClick={false}
          defaultEdgeOptions={{ interactionWidth: 20 }}
          onNodesChange={onNodesChange}
          onNodeDragStop={(_e, node) =>
            void updateItem(node.id, { x: node.position.x, y: node.position.y })
          }
          onNodesDelete={(deleted) => deleted.forEach((n) => void deleteItem(n.id))}
          onConnect={onConnect}
          onEdgesDelete={(deleted) => deleted.forEach((e) => void disconnect(e.id))}
          proOptions={{ hideAttribution: true }}
          minZoom={0.2}
          maxZoom={2}
          fitView
        >
          <Background variant={BackgroundVariant.Dots} gap={16} size={1} className="opacity-40" />
          <Controls showInteractive={false} className="!bottom-3 !left-3" />
        </ReactFlow>
      </BoardActionsContext.Provider>
    </div>
  )
}

export function TrainerCanvas(): React.JSX.Element {
  return (
    <ReactFlowProvider>
      <Canvas />
    </ReactFlowProvider>
  )
}
