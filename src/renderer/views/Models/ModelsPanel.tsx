/**
 * A read-only view of every models root on disk. No add, delete, or rename: this panel answers
 * "what does Core actually see", which is otherwise only visible as filenames inside a node's
 * model dropdown.
 *
 * More than one root shows up when `--models-dir` (or `INLINE_MODELS_DIR`) points somewhere other
 * than `./models`; only the first is written to by downloads and training.
 */
import { useEffect, useState } from 'react'
import type { ModelTreeDir, ModelTreeNode, ModelTreeRoot } from '@shared/types'
import { useModelsTreeStore } from '../../store/modelsTreeStore'
import { useCoreNodesStore } from '../../store/coreNodesStore'
import { ChevronDownIcon, ChevronRightIcon, FolderIcon, ModelsIcon } from '../../components/icons'

function formatSize(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`
  if (bytes >= 1024 ** 2) return `${Math.round(bytes / 1024 ** 2)} MB`
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`
  return `${bytes} B`
}

function countFiles(node: ModelTreeNode): number {
  if (node.kind === 'file') return 1
  return node.children.reduce((sum, child) => sum + countFiles(child), 0)
}

function TreeNode({ node, depth }: { node: ModelTreeNode; depth: number }): React.JSX.Element {
  // Everything starts collapsed: a full models root is hundreds of files, so the panel opens on
  // the category list and the user drills in from there.
  const [open, setOpen] = useState(false)

  if (node.kind === 'file') {
    return (
      <div
        className="flex items-center gap-1.5 py-0.5 pr-2"
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
        title={node.path}
      >
        <ModelsIcon className="h-3 w-3 shrink-0 text-zinc-600" />
        <span className="min-w-0 flex-1 truncate text-[11px] text-zinc-300">{node.name}</span>
        <span className="shrink-0 text-[10px] tabular-nums text-zinc-600">
          {formatSize(node.sizeBytes)}
        </span>
      </div>
    )
  }

  return (
    <div className="flex flex-col">
      <button
        onClick={() => setOpen((o) => !o)}
        style={{ paddingLeft: `${depth * 12 + 4}px` }}
        className="flex items-center gap-1 py-0.5 text-left text-[10px] uppercase tracking-wide text-zinc-500 hover:text-zinc-300"
        title={node.path}
      >
        {open ? (
          <ChevronDownIcon className="h-3 w-3 shrink-0" />
        ) : (
          <ChevronRightIcon className="h-3 w-3 shrink-0" />
        )}
        <FolderIcon className="h-3 w-3 shrink-0 text-zinc-500" />
        <span className="truncate">{node.name}</span>
        <span className="text-zinc-600">({countFiles(node)})</span>
      </button>
      {open && (
        <div className="flex flex-col">
          {node.children.map((child) => (
            <TreeNode key={child.path} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  )
}

function Root({ root, showPath }: { root: ModelTreeRoot; showPath: boolean }): React.JSX.Element {
  const total = root.categories.reduce((sum, c) => sum + countFiles(c), 0)
  return (
    <div className="border-b border-border pb-2 last:border-b-0">
      {showPath && (
        <div className="px-2 pb-1 pt-2">
          <div className="flex items-center gap-1.5">
            <span className="min-w-0 flex-1 truncate text-[11px] font-medium text-zinc-300">
              {root.label}
            </span>
            {!root.writable && (
              <span className="shrink-0 rounded bg-zinc-700 px-1 text-[9px] text-zinc-300">
                read only
              </span>
            )}
          </div>
          <div className="truncate text-[10px] text-zinc-600" title={root.path}>
            {root.path}
          </div>
        </div>
      )}
      {total === 0 ? (
        <p className="px-2 py-2 text-[11px] text-zinc-600">No weight files here yet.</p>
      ) : (
        root.categories.map((category: ModelTreeDir) => (
          <TreeNode key={category.path} node={category} depth={0} />
        ))
      )}
    </div>
  )
}

export function ModelsPanel(): React.JSX.Element {
  const roots = useModelsTreeStore((s) => s.roots)
  const loading = useModelsTreeStore((s) => s.loading)
  const error = useModelsTreeStore((s) => s.error)
  const load = useModelsTreeStore((s) => s.load)
  const registryVersion = useCoreNodesStore((s) => s.registryVersion)

  // registryVersion changes whenever a weight file lands, so this doubles as the refresh trigger.
  useEffect(() => {
    void load(registryVersion ?? undefined)
  }, [load, registryVersion])

  return (
    <div className="flex h-full flex-col bg-surface">
      <div className="flex items-center justify-between border-b border-border px-2 py-1.5">
        <span className="text-[11px] font-medium text-zinc-300">Models on disk</span>
        <button
          onClick={() => void load()}
          className="rounded px-1.5 py-0.5 text-[10px] text-zinc-500 hover:bg-black/30 hover:text-zinc-300"
        >
          Refresh
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {error && <p className="px-2 py-3 text-[11px] text-red-400">{error}</p>}
        {!error && loading && roots.length === 0 && (
          <p className="px-2 py-3 text-[11px] text-zinc-500">Scanning…</p>
        )}
        {!error && !loading && roots.length === 0 && (
          <p className="px-2 py-3 text-[11px] text-zinc-500">No models root is configured.</p>
        )}
        {roots.map((root) => (
          <Root key={root.path} root={root} showPath={roots.length > 1} />
        ))}
      </div>
    </div>
  )
}
