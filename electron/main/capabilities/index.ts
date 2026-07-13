/** The active native capabilities, bound once at boot by whichever shell is running. */
import type { Capabilities } from './types'

export * from './types'

let current: Capabilities | null = null

/** Bind the shell capabilities. Call before registerIpcHandlers(). */
export function setCapabilities(next: Capabilities): void {
  current = next
}

/** The active capabilities. Throws if no shell has set them yet. */
export function caps(): Capabilities {
  if (!current) throw new Error('Capabilities not set. Call setCapabilities() at boot.')
  return current
}
