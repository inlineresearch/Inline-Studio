/**
 * The recent-projects list, persisted as a small JSON file in Electron's
 * userData dir (app-global, not per-project).
 */
import { join } from 'node:path'
import { readFileSync, writeFileSync, existsSync } from 'node:fs'
import type { RecentProject } from '@shared/types'
import { caps } from '../capabilities'

const MAX_RECENTS = 12

function recentsFile(): string {
  return join(caps().appDataDir(), 'recent-projects.json')
}

export function listRecents(): RecentProject[] {
  const file = recentsFile()
  if (!existsSync(file)) return []
  try {
    const parsed = JSON.parse(readFileSync(file, 'utf-8')) as RecentProject[]
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function recordRecent(entry: { name: string; path: string }): void {
  const now = Date.now()
  const existing = listRecents().filter((r) => r.path !== entry.path)
  const next: RecentProject[] = [
    { name: entry.name, path: entry.path, lastOpenedAt: now },
    ...existing,
  ].slice(0, MAX_RECENTS)
  writeFileSync(recentsFile(), JSON.stringify(next, null, 2), 'utf-8')
}
