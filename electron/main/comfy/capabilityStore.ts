/**
 * Persisted, TTL-cached ComfyUI capability snapshots, keyed by the ComfyUI URL so
 * switching installs (incl. ephemeral cloud boxes) stays clean. Capabilities describe the
 * machine/install — not the project — so they live in userData, like settings/credentials.
 */
import { join } from 'node:path'
import { readFileSync, writeFileSync, existsSync } from 'node:fs'
import { caps } from '../capabilities'
import { getSettings } from '../settings/store'
import { fetchCapabilities, type ComfyCapabilities } from './client'

const DEFAULT_MAX_AGE_MS = 10 * 60 * 1000

function cacheFile(): string {
  return join(caps().appDataDir(), 'comfy-capabilities.json')
}

function urlKey(): string {
  return getSettings().comfyUrl.replace(/\/+$/, '')
}

function readAll(): Record<string, ComfyCapabilities> {
  try {
    if (existsSync(cacheFile())) {
      return JSON.parse(readFileSync(cacheFile(), 'utf-8')) as Record<string, ComfyCapabilities>
    }
  } catch {
    // fall through to empty
  }
  return {}
}

function writeAll(all: Record<string, ComfyCapabilities>): void {
  try {
    writeFileSync(cacheFile(), JSON.stringify(all))
  } catch {
    // best-effort cache; a write failure just means we refetch next time
  }
}

export function getCachedCapabilities(url = urlKey()): ComfyCapabilities | null {
  return readAll()[url] ?? null
}

/** Fresh-enough cached capabilities, else fetch + persist. `force` always refetches. */
export async function getCapabilities(opts?: {
  maxAgeMs?: number
  force?: boolean
}): Promise<ComfyCapabilities> {
  const url = urlKey()
  const cached = getCachedCapabilities(url)
  const maxAge = opts?.maxAgeMs ?? DEFAULT_MAX_AGE_MS
  if (!opts?.force && cached && Date.now() - cached.fetchedAt < maxAge) return cached
  const fresh = await fetchCapabilities()
  const all = readAll()
  all[fresh.url] = fresh
  writeAll(all)
  return fresh
}
