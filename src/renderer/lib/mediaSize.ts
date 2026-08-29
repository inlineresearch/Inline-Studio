// Core stores no dimensions on an asset, so an aspect ratio has to be decoded in the browser.

/** Give up rather than hang a caller on a file whose metadata never arrives. */
const MEASURE_TIMEOUT_MS = 8000

export function imageSize(url: string): Promise<{ w: number; h: number } | null> {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => resolve({ w: img.naturalWidth, h: img.naturalHeight })
    img.onerror = () => resolve(null)
    img.src = url
  })
}

export function videoSize(url: string): Promise<{ w: number; h: number } | null> {
  return new Promise((resolve) => {
    const v = document.createElement('video')
    v.preload = 'metadata'
    v.muted = true
    v.onloadedmetadata = () =>
      resolve(v.videoWidth && v.videoHeight ? { w: v.videoWidth, h: v.videoHeight } : null)
    v.onerror = () => resolve(null)
    v.src = url
  })
}

/** The media's width/height ratio, or null when it can't be decoded (audio always). */
export async function mediaAspect(url: string, kind: string): Promise<number | null> {
  if (kind !== 'image' && kind !== 'video') return null
  const timeout = new Promise<null>((resolve) =>
    setTimeout(() => resolve(null), MEASURE_TIMEOUT_MS),
  )
  const size = await Promise.race([kind === 'video' ? videoSize(url) : imageSize(url), timeout])
  return size && size.w > 0 && size.h > 0 ? size.w / size.h : null
}
