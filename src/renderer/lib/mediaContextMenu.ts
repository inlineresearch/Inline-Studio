/**
 * Right-click actions for project media on a surface (frame, preview node, asset
 * thumb). Images get "Copy image" (to the system clipboard); video/audio get a
 * "Save…" action (they can't be copied as an image). Returns an `onContextMenu`
 * handler that opens the shared context menu.
 */
import type { AssetKind } from '@shared/types'
import { studio } from '@/lib/studio'
import { useContextMenuStore, type ContextMenuItem } from '../store/contextMenuStore'
import { ipcErrorMessage } from './ipcError'

/** Copy a project image (by media URL or relative path) to the system clipboard. */
async function copyImage(src: string): Promise<void> {
  try {
    const res = await studio().media.copyImage(src)
    if (!res.ok) console.error('Copy image failed:', res.error)
  } catch (e) {
    console.error('Copy image failed:', ipcErrorMessage(e))
  }
}

/** Save a project media file (by media URL or relative path) to a user-chosen location. */
async function saveMedia(src: string, suggestedName: string): Promise<void> {
  try {
    const res = await studio().media.save(src, suggestedName)
    if (!res.ok) console.error('Save media failed:', res.error)
  } catch (e) {
    console.error('Save media failed:', ipcErrorMessage(e))
  }
}

interface MediaRef {
  src: string
  name: string
  kind: AssetKind
}

/** Hook: returns an `onContextMenu` handler for a media element. */
export function useMediaContextMenu(): (e: React.MouseEvent, media: MediaRef) => void {
  const open = useContextMenuStore((s) => s.open)
  return (e, media) => {
    const items: ContextMenuItem[] =
      media.kind === 'image'
        ? [{ label: 'Copy image', onClick: () => void copyImage(media.src) }]
        : [
            {
              label: media.kind === 'video' ? 'Save video…' : 'Save audio…',
              onClick: () => void saveMedia(media.src, media.name),
            },
          ]
    open(e, items)
  }
}
