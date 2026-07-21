import { useEffect, useMemo, useRef, useState } from 'react'
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  ConnectionMode,
  SelectionMode,
  useNodesState,
  useEdgesState,
  useReactFlow,
  useUpdateNodeInternals,
  applyNodeChanges,
  type Node,
  type NodeProps,
  type Edge,
  type Connection,
  type FinalConnectionState,
  type NodeChange,
  type NodeTypes,
  type EdgeTypes,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { studio } from '@/lib/studio'
import { resolveMedia } from '@/lib/media'
import { audioPeaksPath } from '@shared/media'
import { importFilesToLibrary, importMediaUrlToLibrary } from '@/lib/importFiles'
import { copyText } from '@/lib/clipboard'
import type { MoodboardItem, MoodboardConnector, TextItemData, Frame, Asset } from '@shared/types'
import { portKindColor, portsSatisfy, type NodeDescriptor, type PortKind } from '@shared/coreNodes'
import { useMoodboardStore } from '../../store/moodboardStore'
import { useCoreNodesStore } from '../../store/coreNodesStore'
import { useGraphSelectionStore } from '../../store/graphSelectionStore'
import { expandToGraphs, runTargets, toEdges } from './graphSelection'
import { useAssetStore } from '../../store/assetStore'
import { useFrameStore } from '../../store/frameStore'
import { useGenerationStore } from '../../store/generationStore'
import { useModelRequirementsStore } from '../../store/modelRequirementsStore'
import { useTimelineStore } from '../../store/timelineStore'
import { useUiStore } from '../../store/uiStore'
import {
  getAssetDragIds,
  getFrameDragId,
  getMediaFileDrag,
  getOutputDragId,
  getOutputTakeId,
} from '../../lib/dnd'
import { ImageNode } from './nodes/ImageNode'
import { VideoNode } from './nodes/VideoNode'
import { AudioNode } from './nodes/AudioNode'
import { TextNode } from './nodes/TextNode'
import { FrameNode } from './nodes/FrameNode'
import { GenNode } from './nodes/GenNode'
import { PromptNode } from './nodes/PromptNode'
import { GenerateSettingsPanel } from './GenerateSettingsPanel'
import { CoreSettingsPanel } from './CoreSettingsPanel'
import { ModelInfoPanel } from './ModelInfoPanel'
import { ModelRequirementsModal } from './nodes/ModelRequirementsModal'
import { PreviewNode } from './nodes/PreviewNode'
import { LayerNode } from './nodes/LayerNode'
import { DirectorNode } from './nodes/DirectorNode'
import { TrimNode } from './nodes/TrimNode'
import { LoaderNode } from './nodes/LoaderNode'
import { GraphNode } from './nodes/GraphNode'
import { DeletableEdge } from './edges/DeletableEdge'
import { SideMenu } from './SideMenu'
import { CanvasToolbar } from './CanvasToolbar'
import { AddNodeMenu, type AddNodeKind } from './AddNodeMenu'
import { FrameInspector } from './FrameInspector'
import { CheckIcon, CloseIcon, CopyIcon } from '../../components/icons'

/**
 * A `type:'frame'` canvas item renders as the self-contained fal GenNode when its backing frame is
 * `provider:'fal'`; anything else (a legacy `unset`/`comfy` frame, or an old loader frame) falls
 * through to the plain FrameNode viewer. Fal nodes are added directly from the Add-node dropdown.
 */
function FrameNodeSwitch(props: NodeProps): React.JSX.Element {
  const { frameId, loader } = props.data as { frameId: string; loader?: boolean }
  const provider = useFrameStore((s) => s.frames.find((f) => f.id === frameId)?.provider)
  if (!loader && provider === 'fal') return <GenNode {...props} />
  return <FrameNode {...props} />
}

const nodeTypes: NodeTypes = {
  image: ImageNode,
  video: VideoNode,
  audio: AudioNode,
  text: TextNode,
  frame: FrameNodeSwitch,
  preview: PreviewNode,
  layer: LayerNode,
  director: DirectorNode,
  trim: TrimNode,
  prompt: PromptNode,
  loader: LoaderNode,
  core: GraphNode,
}

const edgeTypes: EdgeTypes = {
  deletable: DeletableEdge,
}

// Visual frame-link colors by chain level: first hop (from a root frame) = light red,
// next hop a different color, and so on, cycling through the palette.
const LEVEL_COLORS = [
  '#fca5a5', // red-300
  '#fdba74', // orange-300
  '#fcd34d', // amber-300
  '#86efac', // green-300
  '#93c5fd', // blue-300
  '#d8b4fe', // purple-300
  '#f9a8d4', // pink-300
]

const isFunctionalConnector = (c: MoodboardConnector): boolean => {
  const s = (c.data?.sourceHandle as string | undefined) ?? 'out'
  const t = (c.data?.targetHandle as string | undefined) ?? 'in'
  return s === 'out' && t === 'in'
}

/**
 * Color each visual (frame↔frame) connector by its level: a connector's color is
 * the depth of its source node in the link graph (roots = depth 0 → level-1
 * links are all light red; their children's links the next color, etc.).
 */
function visualEdgeColors(connectors: MoodboardConnector[]): Map<string, string> {
  const visual = connectors.filter((c) => !isFunctionalConnector(c))
  const adj = new Map<string, string[]>()
  const targets = new Set<string>()
  const sources = new Set<string>()
  for (const c of visual) {
    if (!adj.has(c.fromItemId)) adj.set(c.fromItemId, [])
    adj.get(c.fromItemId)?.push(c.toItemId)
    targets.add(c.toItemId)
    sources.add(c.fromItemId)
  }
  const depth = new Map<string, number>()
  const queue: string[] = []
  const visit = (id: string, d: number): void => {
    if (depth.has(id)) return
    depth.set(id, d)
    queue.push(id)
  }
  // Roots = sources with no incoming visual link.
  for (const s of sources) if (!targets.has(s)) visit(s, 0)
  // All-cycle fallback so every link still gets a color.
  if (queue.length === 0) for (const s of sources) visit(s, 0)
  while (queue.length) {
    const n = queue.shift() as string
    const d = depth.get(n) ?? 0
    for (const t of adj.get(n) ?? []) visit(t, d + 1)
  }
  const colors = new Map<string, string>()
  for (const c of visual) {
    const d = depth.get(c.fromItemId) ?? 0
    colors.set(c.id, LEVEL_COLORS[d % LEVEL_COLORS.length])
  }
  return colors
}

/** The port kind of a core node's handle, resolved from its descriptor (null for non-core/unknown). */
function corePortKindOf(
  coreType: string | undefined,
  handle: string,
  side: 'input' | 'output',
  descriptorsByType: Map<string, NodeDescriptor>,
): PortKind | null {
  if (!coreType) return null
  const descriptor = descriptorsByType.get(coreType)
  const ports = side === 'output' ? descriptor?.outputs : descriptor?.inputs
  return ports?.find((p) => p.id === handle)?.kind ?? null
}

/**
 * The wire color for a connector between two low-level Core ports - the source dot's kind color so
 * the link matches the dots it joins (model = red, vae = orange, latent = blue, …). Falls back to
 * the target's kind (same kind by the type rule) and finally undefined for non-core links, which keep
 * the frame-flow visual coloring.
 */
function edgeKindColor(
  from: string,
  to: string,
  sourceHandle: string,
  targetHandle: string,
  coreTypeById: Map<string, string>,
  descriptorsByType: Map<string, NodeDescriptor>,
): string | undefined {
  const kind =
    corePortKindOf(coreTypeById.get(from), sourceHandle, 'output', descriptorsByType) ??
    corePortKindOf(coreTypeById.get(to), targetHandle, 'input', descriptorsByType)
  return kind ? portKindColor(kind) : undefined
}

/** Parse the slot index from a director handle id (e.g. "vin-3" → 3), or null. */
function slotIndex(handle: string | undefined, prefix: string): number | null {
  if (typeof handle !== 'string' || !handle.startsWith(prefix)) return null
  const n = Number(handle.slice(prefix.length))
  return Number.isFinite(n) ? n : null
}

/** The media kind feeding a director input (frame/asset/preview/trim source). */
function directorInputKind(
  src: MoodboardItem | undefined,
  items: MoodboardItem[],
  connectors: MoodboardConnector[],
  frames: Frame[],
  assets: Asset[],
  depth = 0,
): 'image' | 'video' | 'audio' {
  if (!src || depth > 8) return 'video'
  if (src.type === 'asset' && src.assetId) {
    return assets.find((a) => a.id === src.assetId)?.kind ?? 'video'
  }
  if (src.type === 'frame' && src.frameId) {
    return frames.find((f) => f.id === src.frameId)?.kind ?? 'video'
  }
  if (src.type === 'preview') {
    const feed = connectors.find((k) => k.toItemId === src.id)
    const ff = feed ? items.find((it) => it.id === feed.fromItemId) : undefined
    return ff?.frameId ? (frames.find((f) => f.id === ff.frameId)?.kind ?? 'video') : 'video'
  }
  if (src.type === 'trim') {
    // Follow the trim node's input to the real source kind.
    const feed = connectors.find((k) => k.toItemId === src.id)
    const up = feed ? items.find((it) => it.id === feed.fromItemId) : undefined
    return directorInputKind(up, items, connectors, frames, assets, depth + 1)
  }
  return 'video'
}

const FALLBACK_TEXT: TextItemData = {
  text: '',
  fontSize: 18,
  bold: false,
  italic: false,
  underline: false,
  color: '#e4e4e7',
  align: 'left',
}

/** The unified node canvas ("Inline Studio"): frames, layers, previews, and ideation items. */
export function MoodboardPanel(): React.JSX.Element {
  return (
    <ReactFlowProvider>
      <Board />
    </ReactFlowProvider>
  )
}

function Board(): React.JSX.Element {
  const { items, connectors, error, load, updateItem, deleteItem, connect, disconnect } =
    useMoodboardStore()
  const addTextAt = useMoodboardStore((s) => s.addTextAt)
  const addFrameFromAssetInLayer = useMoodboardStore((s) => s.addFrameFromAssetInLayer)
  const addFrameItemInLayer = useMoodboardStore((s) => s.addFrameItemInLayer)
  const addPreview = useMoodboardStore((s) => s.addPreview)
  const addLayer = useMoodboardStore((s) => s.addLayer)
  const addDirector = useMoodboardStore((s) => s.addDirector)
  const addTrim = useMoodboardStore((s) => s.addTrim)
  const addEmptyFrame = useMoodboardStore((s) => s.addEmptyFrame)
  const addLoader = useMoodboardStore((s) => s.addLoader)
  const addLoaderAssets = useMoodboardStore((s) => s.addLoaderAssets)
  const addPrompt = useMoodboardStore((s) => s.addPrompt)
  const addCoreNode = useMoodboardStore((s) => s.addCoreNode)
  const addGenNode = useMoodboardStore((s) => s.addGenNode)
  const duplicateItems = useMoodboardStore((s) => s.duplicateItems)
  const undo = useMoodboardStore((s) => s.undo)
  const redo = useMoodboardStore((s) => s.redo)
  const addSourceInput = useFrameStore((s) => s.addSourceInput)
  const setHero = useFrameStore((s) => s.setHero)
  const genError = useGenerationStore((s) => s.error)
  const setGenError = useGenerationStore((s) => s.setError)
  const frames = useFrameStore((s) => s.frames)
  const assets = useAssetStore((s) => s.assets)
  const coreDescriptors = useCoreNodesStore((s) => s.descriptors)
  // Load the Inline Core node palette once; descriptors drive the low-level graph nodes.
  useEffect(() => {
    void useCoreNodesStore.getState().load()
  }, [])
  const loadAssets = useAssetStore((s) => s.load)
  const loadFrames = useFrameStore((s) => s.load)
  const setCanvasSelection = useUiStore((s) => s.setCanvasSelection)
  const setCanvasCenter = useUiStore((s) => s.setCanvasCenter)
  const wrapperRef = useRef<HTMLDivElement>(null)
  const { screenToFlowPosition, getNodes } = useReactFlow()
  const updateNodeInternals = useUpdateNodeInternals()
  const [nodes, setNodes] = useNodesState<Node>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([])
  // In-memory clipboard for copy/paste; pasteCount cascades repeated pastes.
  const clipboard = useRef<MoodboardItem[]>([])
  const pasteCount = useRef(0)
  // "Connect to…" menu shown when an output link is dropped on empty canvas.
  const [connectMenu, setConnectMenu] = useState<{
    fromItemId: string
    flowX: number
    flowY: number
    menuX: number
    menuY: number
  } | null>(null)
  // Active canvas tool: Select (edit - marquee/move) vs Pan (view - drag pans, nodes locked).
  const [tool, setTool] = useState<'select' | 'pan'>('select')
  // "Add node" list (toolbar + button, or double-click empty canvas in Select mode).
  const [addMenu, setAddMenu] = useState<{
    x: number
    y: number
    /** Grow upward + center on x (toolbar button) vs top-left at (x,y) (cursor). */
    above: boolean
    flowX: number
    flowY: number
  } | null>(null)

  useEffect(() => {
    void load()
    void loadAssets()
    void loadFrames()
  }, [load, loadAssets, loadFrames])

  const assetsById = useMemo(() => new Map(assets.map((a) => [a.id, a])), [assets])

  // Compact/plumbing core nodes (no media output - loaders, samplers, encoders) hug their content
  // rather than stretch to a stored height, so a Load node is just its title + file dropdown.
  const compactCoreTypes = useMemo(
    () => new Set(coreDescriptors.filter((d) => d.outputKind === null).map((d) => d.type)),
    [coreDescriptors],
  )

  // Lookups for coloring engine wires by their port kind (see edgeKindColor). Keyed by type/id so
  // they only rebuild when the node set or the palette changes, not on every drag frame.
  const descriptorsByType = useMemo(
    () => new Map(coreDescriptors.map((d) => [d.type, d])),
    [coreDescriptors],
  )
  const coreTypeById = useMemo(
    () =>
      new Map(
        items
          .filter((i) => i.type === 'core' && i.data.core)
          .map((i) => [i.id, (i.data.core as { type: string }).type]),
      ),
    [items],
  )

  useEffect(() => {
    setNodes(toNodes(items, assetsById, compactCoreTypes))
  }, [items, assetsById, compactCoreTypes, setNodes])

  // Director render progress (main → renderer) drives the editor + node progress UI.
  useEffect(() => {
    return studio().timeline.onProgress((e) => {
      useTimelineStore.getState().setProgress(e.ownerItemId, e.fraction >= 1 ? null : e.fraction)
    })
  }, [])

  // fal generation lifecycle (main → renderer): drive per-node progress, and refresh takes as
  // nodes complete so their thumbnails, downstream inputs, and the Library all update live.
  useEffect(() => {
    const gen = useGenerationStore.getState()
    // Finish any generations that were still running when the app last closed (they keep
    // running server-side); their progress/completion arrives through the events below.
    void gen.resumePending()
    const unsubs = [
      studio().events.onGenerationProgress((e) => {
        gen.setBusy(e.frameId, true)
        gen.setProgress(e.frameId, e.fraction, e.status)
      }),
      studio().events.onGenerationNodeDone((e) => {
        gen.setBusy(e.frameId, false)
        gen.setProgress(e.frameId, null)
        void useFrameStore.getState().load()
      }),
      studio().events.onGenerationDone(() => {
        gen.finishAll()
        void useFrameStore.getState().load()
        // Core workflow outputs are stored on their canvas items - refresh the board to show them.
        void useMoodboardStore.getState().load()
      }),
      studio().events.onGenerationError((e) => {
        gen.finishAll()
        gen.setError(e.error)
      }),
      // Explicit model downloads (the node's "missing models" popup) stream here.
      studio().events.onModelDownloadProgress((e) => {
        useModelRequirementsStore.getState().onProgress(e)
      }),
      studio().events.onModelDownloadDone((e) => {
        useModelRequirementsStore.getState().onDone(e)
      }),
      studio().events.onModelDownloadError((e) => {
        useModelRequirementsStore.getState().onError(e)
      }),
    ]
    return () => unsubs.forEach((u) => u())
  }, [])

  // `onlyRenderVisibleElements` culls/sizes nodes from their measured dimensions, which
  // come from the pane. While the window is minimized/occluded, the pane measures 0×0, so any node
  // re-measured then gets stuck at zero size and stays invisible after returning - most visibly the
  // preview nodes (frames self-correct via their aspect-fit re-measure). Force a fresh re-measure of
  // every node whenever the canvas becomes visible again.
  useEffect(() => {
    const remeasure = (): void => {
      if (document.visibilityState !== 'visible') return
      const ids = useMoodboardStore.getState().items.map((it) => it.id)
      if (ids.length) updateNodeInternals(ids)
    }
    const raf = requestAnimationFrame(remeasure)
    document.addEventListener('visibilitychange', remeasure)
    window.addEventListener('focus', remeasure)
    return () => {
      cancelAnimationFrame(raf)
      document.removeEventListener('visibilitychange', remeasure)
      window.removeEventListener('focus', remeasure)
    }
  }, [updateNodeInternals])

  // Edges are managed by useEdgesState (so selection/hover changes apply via
  // onEdgesChange) but kept in sync with the persisted connectors.
  useEffect(() => {
    const levelColors = visualEdgeColors(connectors)
    setEdges(
      connectors.map((c) => {
        const sourceHandle = (c.data?.sourceHandle as string | undefined) ?? 'out'
        const targetHandle = (c.data?.targetHandle as string | undefined) ?? 'in'
        // Input/output links render as solid wires; visual frame links are static and
        // colored by their chain level. A GenNode's audio input ('audio') is a data link too.
        const functional =
          sourceHandle === 'out' && (targetHandle === 'in' || targetHandle === 'audio')
        // Engine wires (between low-level Core ports) take the source dot's kind color so the link
        // matches the dots it joins; other links keep the frame-flow coloring.
        const kindColor = edgeKindColor(
          c.fromItemId,
          c.toItemId,
          sourceHandle,
          targetHandle,
          coreTypeById,
          descriptorsByType,
        )
        return {
          id: c.id,
          source: c.fromItemId,
          target: c.toItemId,
          sourceHandle,
          targetHandle,
          type: 'deletable',
          animated: false,
          data: { functional, color: levelColors.get(c.id), kindColor },
        }
      }),
    )
  }, [connectors, coreTypeById, descriptorsByType, setEdges])

  const onNodesChange = (changes: NodeChange<Node>[]): void => {
    setNodes((nds) => applyNodeChanges(changes, nds))
  }

  // Mirror the canvas selection to the UI store so the assistant knows what's selected.
  // Derived from the live nodes state (reflects clicks AND marquee), keyed so it only
  // writes when the selected set actually changes.
  const selectedKey = nodes
    .filter((n) => n.selected)
    .map((n) => n.id)
    .sort()
    .join(',')
  useEffect(() => {
    setCanvasSelection(selectedKey ? selectedKey.split(',') : [])
  }, [selectedKey, setCanvasSelection])

  // Graph-level selection: selecting any node selects its whole connected graph, and the graph's
  // single Run control lives on its output node (the runnable node with nothing runnable
  // downstream). Expands the live selection to the full graph(s), then publishes the run target(s).
  const setRunTargets = useGraphSelectionStore((s) => s.setRunTargets)
  useEffect(() => {
    const seeds = selectedKey ? selectedKey.split(',') : []
    if (seeds.length === 0) {
      setRunTargets([])
      return
    }
    const byId = new Map(items.map((it) => [it.id, it]))
    const edges = toEdges(connectors, (id) => byId.has(id))
    const graph = expandToGraphs(seeds, edges)

    // Pull the selection out to the whole graph (converges: a re-run with the graph already
    // selected makes no change).
    if (graph.size !== seeds.length || !seeds.every((id) => graph.has(id))) {
      setNodes((nds) => nds.map((n) => ({ ...n, selected: graph.has(n.id) })))
    }

    const isRunnable = (id: string): boolean => {
      const it = byId.get(id)
      if (!it) return false
      if (it.type === 'core') {
        const type = it.data.core?.type
        const d = type ? coreDescriptors.find((dd) => dd.type === type) : undefined
        return !!d && d.outputKind != null // a generation core node (loaders have no media output)
      }
      if (it.type === 'frame') {
        const f = it.frameId ? frames.find((fr) => fr.id === it.frameId) : undefined
        return f?.provider === 'fal' // a fal Generate node
      }
      return false
    }
    setRunTargets(runTargets(graph, edges, isRunnable))
  }, [selectedKey, connectors, items, coreDescriptors, frames, setNodes, setRunTargets])

  // Keyboard: copy (⌘/Ctrl-C), paste (⌘/Ctrl-V), undo (⌘/Ctrl-Z), redo (⌘/Ctrl-Shift-Z
  // or ⌘/Ctrl-Y). Delete is handled by React Flow's deleteKeyCode + onNodesDelete.
  // Ignored while editing text/inputs.
  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (!(e.metaKey || e.ctrlKey)) return
      const t = e.target as HTMLElement | null
      if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return
      const key = e.key.toLowerCase()

      if (key === 'z') {
        e.preventDefault()
        if (e.shiftKey) void redo()
        else void undo()
      } else if (key === 'y') {
        e.preventDefault()
        void redo()
      } else if (key === 'c') {
        const selectedIds = new Set(
          getNodes()
            .filter((n) => n.selected)
            .map((n) => n.id),
        )
        const picked = useMoodboardStore.getState().items.filter((it) => selectedIds.has(it.id))
        if (picked.length) {
          clipboard.current = picked
          pasteCount.current = 0
          e.preventDefault()
        }
      } else if (key === 'v') {
        if (clipboard.current.length === 0) return
        e.preventDefault()
        pasteCount.current += 1
        const shift = 32 * pasteCount.current
        void duplicateItems(clipboard.current, { x: shift, y: shift })
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [getNodes, duplicateItems, undo, redo])

  const centre = (): { x: number; y: number } => {
    const rect = wrapperRef.current?.getBoundingClientRect()
    if (!rect) return { x: 0, y: 0 }
    return screenToFlowPosition({ x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 })
  }

  /** The topmost layer whose rectangle contains an absolute flow point (or null). */
  const layerAt = (pos: { x: number; y: number }, exceptId?: string): MoodboardItem | null => {
    const hit = items
      .filter((it) => it.type === 'layer' && it.id !== exceptId)
      .filter(
        (l) => pos.x >= l.x && pos.x <= l.x + l.width && pos.y >= l.y && pos.y <= l.y + l.height,
      )
    return hit.length ? hit[hit.length - 1] : null
  }

  // The port kind of a low-level Core node's handle (null for non-core nodes / unknown handles).
  const corePortKind = (
    itemId: string | null | undefined,
    handle: string | null | undefined,
    side: 'input' | 'output',
  ): PortKind | null => {
    const item = items.find((it) => it.id === itemId)
    const core = item?.type === 'core' ? item.data.core : undefined
    if (!core) return null
    const descriptor = coreDescriptors.find((d) => d.type === core.type)
    const ports = side === 'output' ? descriptor?.outputs : descriptor?.inputs
    return ports?.find((p) => p.id === handle)?.kind ?? null
  }

  // Type-check a wire before it's made: a Prompt node emits text and every other node emits
  // media, so a Prompt may only feed a text ('prompt') input, and a media output may only feed a
  // non-text input. This blocks a Prompt→image/video dot (and an image/video→prompt dot). Between
  // two low-level Core nodes we apply Core's own engine-type rule (model/latent/conditioning/...).
  const isValidConnection = (c: Connection | Edge): boolean => {
    if (!c.source || !c.target || c.source === c.target) return false
    const srcKind = corePortKind(c.source, c.sourceHandle, 'output')
    const tgtKind = corePortKind(c.target, c.targetHandle, 'input')
    if (srcKind && tgtKind && !portsSatisfy(srcKind, tgtKind)) return false
    const sourceIsText = items.find((it) => it.id === c.source)?.type === 'prompt'
    const targetIsText = (c.targetHandle ?? undefined) === 'prompt'
    return sourceIsText === targetIsText
  }

  const onConnect = (c: Connection): void => {
    if (!c.source || !c.target || c.source === c.target) return
    if (!isValidConnection(c)) return
    const src = items.find((it) => it.id === c.source)
    const tgt = items.find((it) => it.id === c.target)

    // Director input: auto-assign the next free slot on the matching layer (by source
    // kind), so wiring grows without the user having to hit a specific tiny handle dot -
    // and the inputs keep increasing one-by-one as they fill.
    if (tgt?.type === 'director') {
      const prefix =
        directorInputKind(src, items, connectors, frames, assets) === 'audio' ? 'ain-' : 'vin-'
      const used = new Set(
        connectors
          .filter((k) => k.toItemId === tgt.id)
          .map((k) => slotIndex(k.data?.targetHandle as string | undefined, prefix))
          .filter((n): n is number => n !== null),
      )
      const explicit = slotIndex(c.targetHandle ?? undefined, prefix)
      let s = explicit !== null && !used.has(explicit) ? explicit : 0
      while (used.has(s)) s++
      void connect(c.source, tgt.id, c.sourceHandle ?? 'out', `${prefix}${s}`)
      return
    }

    void connect(c.source, c.target, c.sourceHandle ?? null, c.targetHandle ?? null)

    // Output → Frame/GenNode input ('in', or a GenNode's 'audio' dot): also wire the data flow-link
    // (the DAG edge). The target frame takes the source's frame - a frame/GenNode directly, or the
    // frame feeding a Preview - as a live input, resolved to that frame's hero take at generate time.
    if (
      tgt?.type === 'frame' &&
      (c.targetHandle === 'in' || c.targetHandle === 'audio') &&
      tgt.frameId
    ) {
      let sourceFrameId: string | undefined
      if (src?.type === 'frame') {
        sourceFrameId = src.frameId ?? undefined
      } else if (src?.type === 'preview') {
        const feed = connectors.find((k) => k.toItemId === src.id)
        sourceFrameId = feed
          ? (items.find((it) => it.id === feed.fromItemId)?.frameId ?? undefined)
          : undefined
      }
      if (sourceFrameId && sourceFrameId !== tgt.frameId) {
        void addSourceInput(tgt.frameId, sourceFrameId)
      }
    }
    // Wiring into a Director node's input handle just persists the connector (above); the
    // node derives its video/audio layers from its connections reactively.
  }

  // Dropping an OUTPUT link on empty canvas suggests what to create next (preview/frame).
  const onConnectEnd = (event: MouseEvent | TouchEvent, state: FinalConnectionState): void => {
    if (state.isValid) return // landed on a real handle → onConnect already wired it
    if (state.fromHandle?.type !== 'source') return // only suggest from output handles
    const fromItemId = state.fromNode?.id
    if (!fromItemId) return
    const pt = 'changedTouches' in event ? event.changedTouches[0] : event
    if (!pt) return
    const point = { x: pt.clientX, y: pt.clientY }
    const flow = screenToFlowPosition(point)
    const rect = wrapperRef.current?.getBoundingClientRect()
    setConnectMenu({
      fromItemId,
      flowX: flow.x,
      flowY: flow.y,
      menuX: rect ? point.x - rect.left : point.x,
      menuY: rect ? point.y - rect.top : point.y,
    })
  }

  // Open the Add-node list from the toolbar's + button - anchored above the button, new nodes
  // land at the canvas centre.
  const openAddFromButton = (buttonRect: DOMRect): void => {
    const rect = wrapperRef.current?.getBoundingClientRect()
    const { x: flowX, y: flowY } = centre()
    const cx = buttonRect.left + buttonRect.width / 2
    setAddMenu({
      x: rect ? cx - rect.left : cx,
      y: rect ? buttonRect.top - rect.top - 8 : buttonRect.top - 8,
      above: true,
      flowX,
      flowY,
    })
  }

  // Double-click empty canvas in Select mode opens the Add list at the cursor; the node is created
  // there. (zoomOnDoubleClick is off, so this doesn't also zoom.)
  const onCanvasDoubleClick = (e: React.MouseEvent): void => {
    if (tool !== 'select') return
    if (!(e.target as HTMLElement).classList.contains('react-flow__pane')) return
    // The double-click also word-selects nearby text (blue highlight) - clear it.
    window.getSelection()?.removeAllRanges()
    const rect = wrapperRef.current?.getBoundingClientRect()
    const flow = screenToFlowPosition({ x: e.clientX, y: e.clientY })
    setAddMenu({
      x: rect ? e.clientX - rect.left : e.clientX,
      y: rect ? e.clientY - rect.top : e.clientY,
      above: false,
      flowX: flow.x,
      flowY: flow.y,
    })
  }

  const onPickAddNode = (kind: AddNodeKind): void => {
    const m = addMenu
    setAddMenu(null)
    if (!m) return
    switch (kind) {
      case 'load':
        void addLoader(m.flowX, m.flowY)
        break
      case 'layer':
        void addLayer(m.flowX, m.flowY)
        break
      case 'preview':
        void addPreview(m.flowX, m.flowY)
        break
      case 'director':
        void addDirector(m.flowX, m.flowY)
        break
      case 'trim':
        void addTrim(m.flowX, m.flowY)
        break
      case 'prompt':
        void addPrompt(m.flowX, m.flowY)
        break
    }
  }

  const onPickCore = (coreType: string): void => {
    const m = addMenu
    setAddMenu(null)
    if (m) void addCoreNode(coreType, m.flowX, m.flowY)
  }

  const onPickGen = (modelId: string): void => {
    const m = addMenu
    setAddMenu(null)
    if (m) void addGenNode(modelId, m.flowX, m.flowY)
  }

  /** Create a preview node and wire the dropped output into it. */
  const suggestPreview = async (): Promise<void> => {
    const m = connectMenu
    setConnectMenu(null)
    if (!m) return
    const item = await addPreview(m.flowX, m.flowY)
    if (item) await connect(m.fromItemId, item.id, 'out', 'in')
  }

  /** Create a downstream frame that takes the dropped output as its input. */
  const suggestFrame = async (): Promise<void> => {
    const m = connectMenu
    setConnectMenu(null)
    if (!m) return
    const item = await addEmptyFrame(m.flowX, m.flowY)
    if (!item) return
    await connect(m.fromItemId, item.id, 'out', 'in')
    // Resolve the source frame (the output's frame, or the frame feeding a preview) and
    // wire it as the new frame's input - its hero take flows in at generate time.
    const fromItem = items.find((it) => it.id === m.fromItemId)
    let sourceFrameId: string | undefined
    if (fromItem?.type === 'frame') {
      sourceFrameId = fromItem.frameId ?? undefined
    } else if (fromItem?.type === 'preview') {
      const feed = connectors.find((k) => k.toItemId === fromItem.id)
      sourceFrameId = feed
        ? (items.find((it) => it.id === feed.fromItemId)?.frameId ?? undefined)
        : undefined
    }
    if (sourceFrameId && item.frameId) await addSourceInput(item.frameId, sourceFrameId)
  }

  /** Create a director node and wire the dropped output into its matching layer. */
  const suggestDirector = async (): Promise<void> => {
    const m = connectMenu
    setConnectMenu(null)
    if (!m) return
    const item = await addDirector(m.flowX, m.flowY)
    if (!item) return
    const src = items.find((it) => it.id === m.fromItemId)
    const handle =
      directorInputKind(src, items, connectors, frames, assets) === 'audio' ? 'ain-0' : 'vin-0'
    await connect(m.fromItemId, item.id, 'out', handle)
  }

  /** Create an "Edit Video/Audio" (trim) node fed by the dropped output. */
  const suggestTrim = async (): Promise<void> => {
    const m = connectMenu
    setConnectMenu(null)
    if (!m) return
    const item = await addTrim(m.flowX, m.flowY)
    if (item) await connect(m.fromItemId, item.id, 'out', 'in')
  }

  const onDrop = (e: React.DragEvent): void => {
    e.preventDefault()
    const drop = screenToFlowPosition({ x: e.clientX, y: e.clientY })

    // A raw media file dragged from the Outputs tab (a Core-node render with no frame) → import it
    // into the Library, then drop a Load Assets loader fed by it at the drop point.
    const media = getMediaFileDrag(e.dataTransfer)
    if (media) {
      void (async () => {
        const asset = await importMediaUrlToLibrary(resolveMedia(media.filePath), media.name)
        if (!asset) return
        // Surface the imported asset in the store, or the loader can't resolve its input thumb
        // (resolveInputThumbs drops any input whose asset isn't loaded) and shows "no media".
        await loadAssets()
        const loader = await addLoader(drop.x, drop.y)
        if (loader) await addLoaderAssets(loader.id, [asset.id])
      })()
      return
    }

    // A generated output dragged from the Outputs tab → create a NEW frame at the drop, fed by
    // that output. Pin the exact take dragged as the source's hero so the new frame shows it.
    const outputFrameId = getOutputDragId(e.dataTransfer)
    if (outputFrameId) {
      const takeId = getOutputTakeId(e.dataTransfer)
      void (async () => {
        if (takeId) await setHero(outputFrameId, takeId)
        const item = await addEmptyFrame(drop.x, drop.y)
        if (item?.frameId) await addSourceInput(item.frameId, outputFrameId)
      })()
      return
    }

    // A frame dragged from the Timeline tab → place its node on the canvas (skip if
    // it's already placed, to avoid two nodes pointing at the same frame).
    const frameId = getFrameDragId(e.dataTransfer)
    if (frameId) {
      if (items.some((it) => it.type === 'frame' && it.frameId === frameId)) return
      const layer = layerAt(drop)
      const x = layer ? drop.x - layer.x : drop.x
      const y = layer ? drop.y - layer.y : drop.y
      void addFrameItemInLayer(frameId, x, y, layer?.id ?? null)
      return
    }

    const ids = getAssetDragIds(e.dataTransfer)
    if (ids.length === 0) {
      // Files dropped from the OS → import into the library, then place as frames.
      const files = Array.from(e.dataTransfer.files ?? [])
      if (files.length > 0) void placeDroppedFiles(files, drop)
      return
    }
    placeAssetsAt(ids, drop)
  }

  /** Place existing library assets as frames at/near a drop point (cascaded). */
  const placeAssetsAt = (assetIds: string[], drop: { x: number; y: number }): void => {
    assetIds.forEach((assetId, i) => {
      const abs = { x: drop.x + i * 24, y: drop.y + i * 24 }
      const layer = layerAt(abs)
      // Children store positions relative to their layer.
      const x = layer ? abs.x - layer.x : abs.x
      const y = layer ? abs.y - layer.y : abs.y
      void addFrameFromAssetInLayer(assetId, x, y, layer?.id ?? null)
    })
  }

  /** Import dropped OS files (paths under Electron, upload in the browser), then place as frames. */
  const placeDroppedFiles = async (
    files: File[],
    drop: { x: number; y: number },
  ): Promise<void> => {
    const assets = await importFilesToLibrary(files, null)
    if (assets.length === 0) return
    await loadAssets() // surface the new media in the Assets panel too
    placeAssetsAt(
      assets.map((a) => a.id),
      drop,
    )
  }

  /** Persist one dropped node's position and (for frames/previews) re-parent into/out of a layer.
   * `recordHistory` is set only for the first node of a multi-drag so the whole move is one undo. */
  const persistNodeDrop = (node: Node, recordHistory: boolean): void => {
    const item = items.find((it) => it.id === node.id)
    if (!item) return

    if (item.type !== 'frame' && item.type !== 'preview') {
      void updateItem(node.id, { x: node.position.x, y: node.position.y }, recordHistory)
      return
    }

    const parent = item.parentId ? items.find((it) => it.id === item.parentId) : undefined
    const abs = parent
      ? { x: parent.x + node.position.x, y: parent.y + node.position.y }
      : { x: node.position.x, y: node.position.y }
    const target = layerAt(abs)
    const newParentId = target?.id ?? null

    if (newParentId !== item.parentId) {
      const x = target ? abs.x - target.x : abs.x
      const y = target ? abs.y - target.y : abs.y
      void updateItem(node.id, { parentId: newParentId, x, y }, recordHistory)
    } else {
      void updateItem(node.id, { x: node.position.x, y: node.position.y }, recordHistory)
    }
  }

  /**
   * On drag stop, persist every dragged node - not just the one under the cursor. React Flow passes
   * the full set of moved nodes as the third arg; ignoring it left the other selected nodes' new
   * positions unsaved, so they snapped back on the next render/reload.
   */
  const onNodeDragStop = (_e: unknown, node: Node, nodes?: Node[]): void => {
    const dragged = nodes && nodes.length > 0 ? nodes : [node]
    dragged.forEach((n, i) => persistNodeDrop(n, i === 0))
  }

  return (
    <div className="relative flex h-full">
      <SideMenu />

      <div
        ref={wrapperRef}
        className="relative flex-1 bg-panel"
        onDoubleClick={onCanvasDoubleClick}
      >
        {error && (
          <div className="absolute left-2 top-2 z-10 rounded bg-red-950/80 px-2 py-1 text-xs text-red-300">
            {error}
          </div>
        )}
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          connectionMode={ConnectionMode.Loose}
          deleteKeyCode={['Backspace', 'Delete']}
          // Figma/Miro multi-select: hold ⌘/Ctrl/Shift and drag to marquee, or
          // ⌘/Ctrl/Shift-click to add to the selection. Partial = touch to select.
          selectionKeyCode={['Meta', 'Control', 'Shift']}
          multiSelectionKeyCode={['Meta', 'Control', 'Shift']}
          selectionMode={SelectionMode.Partial}
          // Tool-driven interaction. Select: left-drag marquee-selects, nodes movable, middle/right
          // still pans. Pan: left-drag anywhere pans, nodes locked (click still selects). Either
          // way double-click opens the Add menu instead of zooming.
          panOnDrag={tool === 'pan' ? true : [1, 2]}
          selectionOnDrag={tool === 'select'}
          nodesDraggable={tool === 'select'}
          zoomOnDoubleClick={false}
          defaultEdgeOptions={{ interactionWidth: 20 }}
          onDrop={onDrop}
          onDragOver={(e) => {
            e.preventDefault()
            e.dataTransfer.dropEffect = 'copy'
          }}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeDragStop={onNodeDragStop}
          onNodesDelete={(deleted) => deleted.forEach((n) => void deleteItem(n.id))}
          onConnect={onConnect}
          onConnectEnd={onConnectEnd}
          isValidConnection={isValidConnection}
          onEdgesDelete={(deleted) => deleted.forEach((e) => void disconnect(e.id))}
          // Mirror the viewport so the assistant can use it as a "place here" spot.
          onMove={() => setCanvasCenter(centre())}
          onInit={() => setCanvasCenter(centre())}
          proOptions={{ hideAttribution: true }}
          minZoom={0.1}
          maxZoom={4}
          // Only mount on-screen nodes/edges. Beyond perf, this keeps the canvas's GPU
          // compositing layer small - with nodes spread far apart, the full layer can
          // exceed the GPU's max texture size and render as grey/blank when scrolling.
          onlyRenderVisibleElements
          fitView
          // Cap the initial fit's zoom so a board with a single small node (e.g. a new project's
          // chooser) lands at a normal 1:1 view instead of blowing up to fill the viewport.
          fitViewOptions={{ maxZoom: 1 }}
        >
          <Background gap={22} size={2.5} color="#525a66" />
        </ReactFlow>

        {items.length === 0 && <EmptyCanvasHint />}

        {genError && <GenErrorToast message={genError} onDismiss={() => setGenError(null)} />}

        {connectMenu && (
          <>
            <div className="absolute inset-0 z-20" onClick={() => setConnectMenu(null)} />
            <div
              className="absolute z-30 flex w-40 flex-col overflow-hidden rounded-md border border-border bg-panel text-xs shadow-xl"
              style={{ left: connectMenu.menuX, top: connectMenu.menuY }}
            >
              <div className="border-b border-border px-2.5 py-1 text-[10px] uppercase tracking-wide text-zinc-500">
                Connect to…
              </div>
              <button
                onClick={() => void suggestPreview()}
                className="px-2.5 py-1.5 text-left text-zinc-200 hover:bg-surface"
              >
                Preview node
              </button>
              <button
                onClick={() => void suggestFrame()}
                className="px-2.5 py-1.5 text-left text-zinc-200 hover:bg-surface"
              >
                New frame (input)
              </button>
              <button
                onClick={() => void suggestDirector()}
                className="px-2.5 py-1.5 text-left text-zinc-200 hover:bg-surface"
              >
                Video Director
              </button>
              <button
                onClick={() => void suggestTrim()}
                className="px-2.5 py-1.5 text-left text-zinc-200 hover:bg-surface"
              >
                Edit Video/Audio
              </button>
            </div>
          </>
        )}

        {addMenu && (
          <AddNodeMenu
            x={addMenu.x}
            y={addMenu.y}
            above={addMenu.above}
            coreNodes={coreDescriptors}
            onPick={onPickAddNode}
            onPickCore={onPickCore}
            onPickGen={onPickGen}
            onClose={() => setAddMenu(null)}
          />
        )}

        <CanvasToolbar
          tool={tool}
          onSelectTool={() => setTool('select')}
          onPanTool={() => setTool('pan')}
          onAddText={() => {
            const { x, y } = centre()
            void addTextAt(x, y)
          }}
          onOpenAdd={openAddFromButton}
        />

        <GenerateSettingsPanel />
        <CoreSettingsPanel />
        <ModelInfoPanel />
        <ModelRequirementsModal />
      </div>

      <FrameInspector />
    </div>
  )
}

/**
 * Floating toast for a fal generation error. The message can be long (fal often echoes the raw
 * response), so it wraps + scrolls and offers a one-click copy for pasting into a bug report.
 */
function GenErrorToast({
  message,
  onDismiss,
}: {
  message: string
  onDismiss: () => void
}): React.JSX.Element {
  const [copied, setCopied] = useState(false)
  const copy = (): void => {
    void copyText(message).then((ok) => {
      if (!ok) return
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }
  return (
    <div className="absolute left-1/2 top-3 z-30 flex max-w-lg -translate-x-1/2 items-start gap-2 rounded-md border border-red-500/40 bg-red-950/90 px-3 py-2 text-xs text-red-200 shadow-lg">
      <span className="max-h-40 min-w-0 flex-1 overflow-y-auto whitespace-pre-wrap break-words">
        {message}
      </span>
      <button
        onClick={copy}
        title={copied ? 'Copied' : 'Copy error'}
        aria-label="Copy error"
        className={`flex h-5 w-5 shrink-0 items-center justify-center rounded ${
          copied ? 'text-emerald-300' : 'text-red-300 hover:text-red-100'
        }`}
      >
        {copied ? <CheckIcon className="h-3.5 w-3.5" /> : <CopyIcon className="h-3.5 w-3.5" />}
      </button>
      <button
        onClick={onDismiss}
        aria-label="Dismiss"
        className="flex h-5 w-5 shrink-0 items-center justify-center rounded text-red-300 hover:text-red-100"
      >
        <CloseIcon className="h-3.5 w-3.5" />
      </button>
    </div>
  )
}

/** Centered hint shown over an empty canvas. Non-interactive so it never blocks drops. */
function EmptyCanvasHint(): React.JSX.Element {
  return (
    <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
      <div className="flex max-w-sm flex-col items-center gap-2 text-center">
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="h-9 w-9 text-zinc-600"
        >
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <circle cx="8.5" cy="8.5" r="1.5" />
          <path d="m21 15-4.5-4.5L7 20" />
        </svg>
        <p className="text-sm font-medium text-zinc-300">Your canvas is empty</p>
        <p className="text-xs leading-relaxed text-zinc-500">
          Drag an asset from the Assets panel onto the canvas to create your first frame.
        </p>
      </div>
    </div>
  )
}

/** Map items to React Flow nodes - layers first so they precede their children. */
function toNodes(
  items: MoodboardItem[],
  assetsById: Map<
    string,
    { filePath: string; kind: string; name: string; thumbPath?: string | null }
  >,
  compactCoreTypes: Set<string>,
): Node[] {
  const ordered = [...items].sort(
    (a, b) => (a.type === 'layer' ? -1 : 0) - (b.type === 'layer' ? -1 : 0),
  )
  return ordered.map((item) => itemToNode(item, assetsById, compactCoreTypes))
}

function itemToNode(
  item: MoodboardItem,
  assetsById: Map<
    string,
    { filePath: string; kind: string; name: string; thumbPath?: string | null }
  >,
  compactCoreTypes: Set<string>,
): Node {
  // A compact/plumbing core node auto-sizes its height (hugs content) instead of taking the stored
  // height; everything else keeps its persisted box.
  const compact =
    item.type === 'core' && !!item.data.core && compactCoreTypes.has(item.data.core.type)
  const common: Node = {
    id: item.id,
    position: { x: item.x, y: item.y },
    style: {
      width: item.width,
      height: compact ? undefined : item.height,
      zIndex: item.zIndex,
    },
    data: {},
    ...(item.parentId ? { parentId: item.parentId } : {}),
  }
  if (item.type === 'layer') {
    return {
      ...common,
      type: 'layer',
      dragHandle: '.drag-handle',
      data: { name: item.data.name ?? 'Layer', color: item.data.color },
    }
  }
  if (item.type === 'frame') {
    return {
      ...common,
      type: 'frame',
      data: { frameId: item.frameId, loader: !!item.data.loader },
    }
  }
  if (item.type === 'preview') {
    return { ...common, type: 'preview', data: {} }
  }
  if (item.type === 'director') {
    return {
      ...common,
      type: 'director',
      data: { name: item.data.name ?? 'Director', previewUrl: item.data.directorPreview },
    }
  }
  if (item.type === 'trim') {
    return { ...common, type: 'trim', data: {} }
  }
  if (item.type === 'prompt') {
    return { ...common, type: 'prompt', data: {} }
  }
  if (item.type === 'loader') {
    return { ...common, type: 'loader', data: { itemId: item.id } }
  }
  if (item.type === 'core') {
    return { ...common, type: 'core', data: { itemId: item.id } }
  }
  if (item.type === 'text') {
    return { ...common, type: 'text', data: { text: item.data.text ?? FALLBACK_TEXT } }
  }
  const asset = item.assetId ? assetsById.get(item.assetId) : undefined
  // Images render from their downscaled thumbnail when available (full-res only in the
  // Library/Preview); video/audio keep their own source.
  const src = asset
    ? resolveMedia(asset.kind === 'image' ? (asset.thumbPath ?? asset.filePath) : asset.filePath)
    : ''
  const type = asset?.kind === 'video' ? 'video' : asset?.kind === 'audio' ? 'audio' : 'image'
  // Derived from the asset id, not `thumbPath` (which is null for audio imports): Core builds the
  // peaks JSON on first request, so already-imported audio gets a waveform too.
  const waveform =
    asset?.kind === 'audio' && item.assetId ? resolveMedia(audioPeaksPath(item.assetId)) : undefined
  return { ...common, type, data: { src, name: asset?.name ?? '', waveform } }
}
