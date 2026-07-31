/**
 * The Trainer tab's node canvas: Load Dataset → Caption → Trainer → Graph, plus the utility
 * Resource node. Same React Flow + node-card design as the Studio moodboard, but backed by the
 * `trainer` surface so the two boards stay separate.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Background,
  BackgroundVariant,
  ConnectionMode,
  ReactFlow,
  ReactFlowProvider,
  applyNodeChanges,
  useReactFlow,
  type Edge,
  type Node,
  type NodeChange,
  type EdgeTypes,
  type NodeTypes,
  type OnConnect,
} from '@xyflow/react'
import type { MoodboardItem } from '@shared/types'
import { studio } from '@/lib/studio'
import { useTrainerBoardStore, type TrainerNodeKind } from '../../store/trainerBoardStore'
import { useModelRequirementsStore } from '../../store/modelRequirementsStore'
import { BoardActionsContext } from '../Moodboard/nodes/boardActions'
import { CanvasToolbar } from '../Moodboard/CanvasToolbar'
import { TrainerDeletableEdge } from './edges/TrainerDeletableEdge'
import { TrainerAddMenu } from './TrainerAddMenu'
import { ModelRequirementsModal } from '../Moodboard/nodes/ModelRequirementsModal'
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

// Clicking a connector selects it and shows the ✕ to unlink, same as the Studio canvas.
const edgeTypes: EdgeTypes = {
  deletable: TrainerDeletableEdge,
}

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
  const [tool, setTool] = useState<'select' | 'pan'>('select')
  const [addMenu, setAddMenu] = useState<{ x: number; y: number; above: boolean } | null>(null)
  const wrapperRef = useRef<HTMLDivElement>(null)
  const rf = useReactFlow()

  useEffect(() => {
    void load()
  }, [load])

  // Model-download progress for the Trainer node's "missing base model" popup. Wired here (not only
  // in MoodboardPanel) because that panel is unmounted while the Trainer tab is showing.
  useEffect(() => {
    const req = useModelRequirementsStore.getState()
    const unsubs = [
      studio().events.onModelDownloadProgress((e) => req.onProgress(e)),
      studio().events.onModelDownloadDone((e) => req.onDone(e)),
      studio().events.onModelDownloadError((e) => req.onError(e)),
    ]
    return () => unsubs.forEach((u) => u())
  }, [])

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
        type: 'deletable',
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

  /** Double-click empty canvas opens the node list there, matching the Studio canvas. */
  const onCanvasDoubleClick = (e: React.MouseEvent): void => {
    if (tool !== 'select') return
    if (!(e.target as HTMLElement).classList.contains('react-flow__pane')) return
    window.getSelection()?.removeAllRanges() // the double-click also word-selects nearby text
    const rect = wrapperRef.current?.getBoundingClientRect()
    setAddMenu({
      x: rect ? e.clientX - rect.left : e.clientX,
      y: rect ? e.clientY - rect.top : e.clientY,
      above: false,
    })
  }

  return (
    <div
      ref={wrapperRef}
      className="relative h-full min-h-0 w-full"
      onDoubleClick={onCanvasDoubleClick}
    >
      {error && (
        <div className="absolute left-1/2 top-3 z-20 -translate-x-1/2 rounded-md bg-red-500/10 px-3 py-1.5 text-xs text-red-400">
          {error}
        </div>
      )}
      <BoardActionsContext.Provider value={boardActions}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          connectionMode={ConnectionMode.Loose}
          deleteKeyCode={['Backspace', 'Delete']}
          zoomOnDoubleClick={false}
          defaultEdgeOptions={{ interactionWidth: 20 }}
          panOnDrag={tool === 'pan' ? true : [1, 2]}
          selectionOnDrag={tool === 'select'}
          nodesDraggable={tool === 'select'}
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
        </ReactFlow>
      </BoardActionsContext.Provider>
      <CanvasToolbar
        tool={tool}
        onSelectTool={() => setTool('select')}
        onPanTool={() => setTool('pan')}
        onOpenAdd={(buttonRect) => {
          const rect = wrapperRef.current?.getBoundingClientRect()
          setAddMenu({
            x: buttonRect.left + buttonRect.width / 2 - (rect?.left ?? 0),
            y: buttonRect.top - 8 - (rect?.top ?? 0),
            above: true,
          })
        }}
      />
      {addMenu && (
        <TrainerAddMenu
          x={addMenu.x}
          y={addMenu.y}
          above={addMenu.above}
          entries={ADDABLE}
          onPick={addAtViewport}
          onClose={() => setAddMenu(null)}
        />
      )}
      <ModelRequirementsModal />
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
