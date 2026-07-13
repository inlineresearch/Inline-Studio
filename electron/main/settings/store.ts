/**
 * App-global settings persisted as JSON in Electron userData: the ComfyUI backend URL and the
 * Inline Core (/v1) engine URL. Defaults come from env (COMFYUI_URL / INLINE_CORE_URL) then localhost.
 */
import { join } from 'node:path'
import { readFileSync, writeFileSync, existsSync } from 'node:fs'
import type { AppSettings } from '@shared/types'
import { caps } from '../capabilities'

const DEFAULT_COMFY_URL = process.env.COMFYUI_URL || 'http://127.0.0.1:8188'
const DEFAULT_CORE_URL = process.env.INLINE_CORE_URL || 'http://127.0.0.1:8848'

function settingsFile(): string {
  return join(caps().appDataDir(), 'settings.json')
}

function read(): Partial<AppSettings> {
  try {
    if (existsSync(settingsFile())) {
      return JSON.parse(readFileSync(settingsFile(), 'utf-8')) as Partial<AppSettings>
    }
  } catch {
    // fall through to defaults
  }
  return {}
}

function nonEmpty(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null
}

export function getSettings(): AppSettings {
  const saved = read()
  return {
    comfyUrl: nonEmpty(saved.comfyUrl) ?? DEFAULT_COMFY_URL,
    coreUrl: nonEmpty(saved.coreUrl) ?? DEFAULT_CORE_URL,
  }
}

function save(next: AppSettings): AppSettings {
  writeFileSync(settingsFile(), JSON.stringify(next, null, 2), 'utf-8')
  return next
}

export function setComfyUrl(url: string): AppSettings {
  return save({ ...getSettings(), comfyUrl: url.trim() || DEFAULT_COMFY_URL })
}

export function setCoreUrl(url: string): AppSettings {
  return save({ ...getSettings(), coreUrl: url.trim() || DEFAULT_CORE_URL })
}
