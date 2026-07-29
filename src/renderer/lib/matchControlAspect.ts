/**
 * Snap a generation node's Width/Height to the aspect of the control map wired into its Control
 * input, so the pose/depth map isn't stretched or cropped at generation time. Resolves the wired
 * source's image, reads its real pixel size, and writes a gen-friendly (multiple-of-64) size back.
 */
import { resolveMedia } from '@/lib/media'
import { useAssetStore } from '../store/assetStore'
import { useMoodboardStore } from '../store/moodboardStore'

const targetHandleOf = (data: Record<string, unknown>): string | undefined =>
  (data as { targetHandle?: string }).targetHandle

/** The project-relative image path feeding a node's Control input, or null. */
function controlSourcePath(nodeId: string): string | null {
  const { items, connectors } = useMoodboardStore.getState()
  const conn = connectors.find(
    (c) => c.toItemId === nodeId && targetHandleOf(c.data) === 'control_image',
  )
  const src = conn ? items.find((i) => i.id === conn.fromItemId) : undefined
  if (!src) return null
  const assets = useAssetStore.getState().assets
  const byId = (id?: string | null): string | undefined =>
    id ? assets.find((a) => a.id === id)?.filePath : undefined

  if (src.type === 'controlSpace') return byId(src.data.controlAssetId) ?? null
  if (src.type === 'core') return src.data.core?.output?.filePath ?? null
  if (src.type === 'loader') return byId((src.data.assetIds ?? [])[0]) ?? null
  if (src.type === 'asset') return byId(src.assetId) ?? null
  return null
}

function imageSize(url: string): Promise<{ w: number; h: number } | null> {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => resolve({ w: img.naturalWidth, h: img.naturalHeight })
    img.onerror = () => resolve(null)
    img.src = url
  })
}

/** A gen-friendly size at the source aspect: long edge ~1216, both edges a multiple of 64, clamped. */
function fitGenSize(w: number, h: number): { width: number; height: number } {
  const scale = 1216 / Math.max(w, h)
  const snap = (n: number): number =>
    Math.max(256, Math.min(2048, Math.round((n * scale) / 64) * 64))
  return { width: snap(w), height: snap(h) }
}

/** Returns the new size on success (also written to the node), or null if nothing is wired/loadable. */
export async function matchControlAspect(
  nodeId: string,
): Promise<{ width: number; height: number } | null> {
  const path = controlSourcePath(nodeId)
  if (!path) return null
  const dims = await imageSize(resolveMedia(path))
  if (!dims) return null
  const size = fitGenSize(dims.w, dims.h)

  const node = useMoodboardStore.getState().items.find((i) => i.id === nodeId)
  const core = node?.data.core
  if (!core) return null
  await useMoodboardStore.getState().updateItem(nodeId, {
    data: { ...node.data, core: { ...core, params: { ...core.params, ...size } } },
  })
  return size
}
