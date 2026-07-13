/**
 * The media-URL seam. Components resolve project-relative media paths through
 * `resolveMedia()` instead of `mediaUrl()` directly. Under Electron it defaults to the
 * `inlinestudio-media://` scheme; the browser shell overrides it via `setMediaResolver`
 * to point at the web server's `GET /media/*` route.
 */
import { mediaUrl } from '@shared/media'

let resolver: (relativePath: string) => string = mediaUrl

/** Override how media paths resolve (e.g. map to the web server's /media route). */
export function setMediaResolver(fn: (relativePath: string) => string): void {
  resolver = fn
}

/** Resolve a project-relative media path to a URL the browser can load. */
export function resolveMedia(relativePath: string): string {
  return resolver(relativePath)
}
