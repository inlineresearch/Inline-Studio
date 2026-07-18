import { useEffect, useState } from 'react'
import { resolveMedia } from '@/lib/media'
import { audioPeaksPath } from '@shared/media'
import type { Asset, AssetFolder } from '@shared/types'
import { useAssetStore, folderPath } from '../../store/assetStore'
import { setAssetDragPayload } from '../../lib/dnd'
import { useMediaContextMenu } from '../../lib/mediaContextMenu'
import { CloseIcon, CreateNewFolderIcon, FolderIcon, PlusIcon } from '../../components/icons'
import { VideoPreview } from '../../components/VideoPreview'
import { AudioPreview } from '../../components/AudioPreview'

/** Left panel: folder navigation + import + a grid of folders and media. */
export function LibraryPanel(): React.JSX.Element {
  const {
    folders,
    assets,
    currentFolderId,
    selectedId,
    loading,
    error,
    load,
    import: importAssets,
    importFiles,
    remove,
    createFolder,
    deleteFolder,
    navigate,
    select,
  } = useAssetStore()
  const [newFolderName, setNewFolderName] = useState<string | null>(null)
  // Multi-selection for dragging several assets at once (⌘/Ctrl-click to toggle).
  const [dragSel, setDragSel] = useState<string[]>([])
  // Highlight while OS files are dragged over the panel.
  const [fileOver, setFileOver] = useState(false)

  // True when a drag carries OS files (Finder/Explorer), not an internal asset/frame drag.
  const isFileDrag = (e: React.DragEvent): boolean =>
    Array.from(e.dataTransfer.types).includes('Files')

  const onDragOver = (e: React.DragEvent): void => {
    if (!isFileDrag(e)) return
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
    if (!fileOver) setFileOver(true)
  }

  const onDrop = (e: React.DragEvent): void => {
    if (!isFileDrag(e)) return
    e.preventDefault()
    setFileOver(false)
    const files = Array.from(e.dataTransfer.files ?? [])
    if (files.length > 0) void importFiles(files)
  }

  const toggleDragSel = (id: string): void =>
    setDragSel((sel) => (sel.includes(id) ? sel.filter((x) => x !== id) : [...sel, id]))

  useEffect(() => {
    void load()
  }, [load])

  const path = folderPath(folders, currentFolderId)
  const subFolders = folders.filter((f) => f.parentId === currentFolderId)
  const folderAssets = assets.filter((a) => a.folderId === currentFolderId)
  const empty = subFolders.length === 0 && folderAssets.length === 0

  const submitNewFolder = (): void => {
    const name = (newFolderName ?? '').trim()
    if (name) void createFolder(name)
    setNewFolderName(null)
  }

  return (
    <div
      className={`relative flex h-full flex-col bg-surface ${
        fileOver ? 'ring-2 ring-inset ring-accent' : ''
      }`}
      onDragOver={onDragOver}
      onDragLeave={(e) => {
        // Only clear when the cursor actually leaves the panel, not child elements.
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setFileOver(false)
      }}
      onDrop={onDrop}
    >
      {fileOver && (
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center bg-accent/5">
          <span className="rounded-md border border-accent bg-panel/90 px-3 py-1.5 text-xs text-accent">
            Drop to import into {path.length > 0 ? path[path.length - 1].name : 'Library'}
          </span>
        </div>
      )}
      <Breadcrumb
        path={path}
        onNavigate={navigate}
        actions={
          <>
            <button
              onClick={() => setNewFolderName('')}
              title="New Folder"
              aria-label="New Folder"
              className="flex h-7 w-7 items-center justify-center rounded text-zinc-400 hover:text-white"
            >
              <CreateNewFolderIcon className="h-4 w-4" />
            </button>
            <button
              onClick={() => void importAssets()}
              disabled={loading}
              title="Import assets"
              aria-label="Import assets"
              className="flex h-7 w-7 items-center justify-center rounded text-zinc-400 hover:text-white disabled:opacity-40"
            >
              <PlusIcon className="h-4 w-4" />
            </button>
          </>
        }
      />

      {error && <p className="px-3 py-2 text-xs text-red-400">{error}</p>}

      <div className="grid flex-1 auto-rows-min grid-cols-2 gap-2 overflow-y-auto p-2">
        {newFolderName !== null && (
          <div className="col-span-2">
            <input
              autoFocus
              value={newFolderName}
              onChange={(e) => setNewFolderName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') submitNewFolder()
                if (e.key === 'Escape') setNewFolderName(null)
              }}
              onBlur={submitNewFolder}
              placeholder="Folder name"
              className="w-full rounded-md border border-accent bg-surface px-2 py-1.5 text-xs text-zinc-100 outline-none placeholder:text-zinc-500"
            />
          </div>
        )}

        {subFolders.map((folder) => (
          <FolderTile
            key={folder.id}
            folder={folder}
            onOpen={() => navigate(folder.id)}
            onDelete={() => void deleteFolder(folder.id)}
          />
        ))}

        {folderAssets.map((asset) => (
          <AssetThumb
            key={asset.id}
            asset={asset}
            selected={asset.id === selectedId}
            dragSelected={dragSel.includes(asset.id)}
            dragIds={dragSel}
            onSelect={() => {
              setDragSel([])
              select(asset.id)
            }}
            onToggleDrag={() => toggleDragSel(asset.id)}
            onDelete={() => void remove(asset.id)}
          />
        ))}

        {empty && newFolderName === null && (
          <div className="col-span-2 flex flex-col items-center justify-center gap-1 p-6 text-center">
            <p className="text-sm text-zinc-500">This folder is empty</p>
            <p className="text-xs text-zinc-600">Import media or create a folder.</p>
          </div>
        )}
      </div>
    </div>
  )
}

function Breadcrumb({
  path,
  onNavigate,
  actions,
}: {
  path: AssetFolder[]
  onNavigate: (id: string | null) => void
  actions?: React.ReactNode
}): React.JSX.Element {
  return (
    <div className="flex items-center gap-1 border-b border-border px-3 py-1.5 text-xs text-zinc-400">
      <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1">
        <button onClick={() => onNavigate(null)} className="hover:text-zinc-200">
          Library
        </button>
        {path.map((f) => (
          <span key={f.id} className="flex items-center gap-1">
            <span className="text-zinc-600">/</span>
            <button onClick={() => onNavigate(f.id)} className="hover:text-zinc-200">
              {f.name}
            </button>
          </span>
        ))}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-1">{actions}</div>}
    </div>
  )
}

function FolderTile({
  folder,
  onOpen,
  onDelete,
}: {
  folder: AssetFolder
  onOpen: () => void
  onDelete: () => void
}): React.JSX.Element {
  return (
    <div className="group relative">
      <button
        onDoubleClick={onOpen}
        onClick={onOpen}
        title={folder.name}
        className="flex w-full flex-col items-center gap-1 rounded-md border border-border bg-surface px-2 py-3 hover:border-zinc-600"
      >
        <FolderIcon className="h-8 w-8 text-zinc-400" />
        <span className="w-full truncate text-center text-[11px] text-zinc-300">{folder.name}</span>
      </button>
      <button
        onClick={(e) => {
          e.stopPropagation()
          onDelete()
        }}
        title="Delete folder (keeps its media)"
        className="absolute right-1 top-1 hidden rounded bg-black/60 p-0.5 text-zinc-300 group-hover:block hover:text-red-400"
      >
        <CloseIcon className="h-3 w-3" />
      </button>
    </div>
  )
}

function AssetThumb({
  asset,
  selected,
  dragSelected,
  dragIds,
  onSelect,
  onToggleDrag,
  onDelete,
}: {
  asset: Asset
  selected: boolean
  dragSelected: boolean
  dragIds: string[]
  onSelect: () => void
  onToggleDrag: () => void
  onDelete: () => void
}): React.JSX.Element {
  const url = resolveMedia(asset.filePath)
  // Grid cells are small - render images from their downscaled thumbnail when present.
  const imageSrc = resolveMedia(
    asset.kind === 'image' ? (asset.thumbPath ?? asset.filePath) : asset.filePath,
  )
  const videoSrc = resolveMedia(asset.previewPath ?? asset.filePath)
  const poster = asset.thumbPath ? resolveMedia(asset.thumbPath) : undefined
  const onContextMenu = useMediaContextMenu()
  return (
    <div className="group relative">
      <button
        onClick={(e) => (e.metaKey || e.ctrlKey ? onToggleDrag() : onSelect())}
        onContextMenu={(e) =>
          onContextMenu(e, {
            src: resolveMedia(asset.filePath),
            name: asset.name,
            kind: asset.kind,
          })
        }
        draggable
        onDragStart={(e) => {
          const ids = dragSelected && dragIds.length > 0 ? dragIds : [asset.id]
          setAssetDragPayload(e.dataTransfer, ids)
        }}
        title={`${asset.name} - drag onto the Frames Sequence (⌘/Ctrl-click to multi-select)`}
        className={`flex w-full flex-col overflow-hidden rounded-md border text-left ${
          dragSelected
            ? 'border-accent ring-1 ring-accent'
            : selected
              ? 'border-accent'
              : 'border-border hover:border-zinc-600'
        }`}
      >
        <div className="flex aspect-video items-center justify-center bg-black/40">
          {asset.kind === 'image' && (
            <img src={imageSrc} alt={asset.name} className="h-full w-full object-cover" />
          )}
          {asset.kind === 'video' && (
            <VideoPreview src={videoSrc} poster={poster} className="h-full w-full object-cover" />
          )}
          {asset.kind === 'audio' && (
            <AudioPreview
              src={url}
              // Derived from the id rather than `thumbPath` (which is null for audio imports):
              // Core builds the peaks JSON on first request, so existing assets get a waveform too.
              waveformUrl={resolveMedia(audioPeaksPath(asset.id))}
              className="h-full w-full"
            />
          )}
        </div>
        <span className="truncate px-1.5 py-1 text-[11px] text-zinc-400">{asset.name}</span>
      </button>
      <button
        onClick={(e) => {
          e.stopPropagation()
          onDelete()
        }}
        title="Remove asset"
        className="absolute right-1 top-1 hidden rounded bg-black/70 p-0.5 text-zinc-300 group-hover:block hover:text-red-400"
      >
        <CloseIcon className="h-3 w-3" />
      </button>
    </div>
  )
}
