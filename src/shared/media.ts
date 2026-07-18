/**
 * Local project media (assets, takes, thumbs) is served to the renderer through a
 * custom privileged scheme instead of file:// - the sandboxed renderer can't read
 * the filesystem directly. Main resolves these URLs against the open project folder.
 *
 *   inlinestudio-media://local/assets/<id>.png
 */
export const MEDIA_SCHEME = 'inlinestudio-media'

/** Build a media URL from a project-relative path (e.g. "assets/abc.png"). */
export function mediaUrl(relativePath: string): string {
  const clean = relativePath.replace(/\\/g, '/').replace(/^\/+/, '')
  const encoded = clean.split('/').map(encodeURIComponent).join('/')
  return `${MEDIA_SCHEME}://local/${encoded}`
}

/** Project-relative path of an audio take's waveform peaks JSON (by convention). */
export function takeWaveformPath(takeId: string): string {
  return `thumbs/take-${takeId}.peaks.json`
}

/**
 * Project-relative path of the waveform peaks JSON for media's *audio* (e.g. a video's
 * embedded audio, used by the director L1 layer), keyed by the source take/asset id.
 */
export function audioPeaksPath(id: string): string {
  return `thumbs/audio-${id}.peaks.json`
}
