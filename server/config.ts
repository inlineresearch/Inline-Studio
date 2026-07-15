/** Server configuration from the environment. Small and explicit, mirroring inline-core's config. */
import { existsSync } from 'node:fs'
import { homedir } from 'node:os'
import { resolve } from 'node:path'

export interface ServerConfig {
  host: string
  port: number
  /** inline-core /v1 base URL the generation handlers call. */
  coreUrl: string
  /** App-global data dir for settings/recents/credentials (the web appDataDir). */
  dataDir: string
  /** Where new projects are created on the server (the browser has no folder picker). */
  workspaceDir: string
  /** Dir holding the built @inlineresearch/ui bundle (index.js + style.css). */
  uiDir: string
  /** inline-core checkout to spawn as a sidecar. */
  coreDir: string
  /** Command (argv) used to launch inline-core when it is not already running. */
  coreCmd: string[]
  coreManaged: boolean
}

function num(value: string | undefined, fallback: number): number {
  const n = Number(value)
  return Number.isFinite(n) && n > 0 ? n : fallback
}

/**
 * Locate the inline-core checkout sitting next to this repo. The folder's casing varies by clone
 * (`Inline-Core` vs `inline-core`) and Linux is case-sensitive, so probe the known names and fall
 * back to the lowercase default when none exist.
 */
function resolveCoreDir(): string {
  const parent = resolve(process.cwd(), '..')
  for (const name of ['Inline-Core', 'inline-core']) {
    const dir = resolve(parent, name)
    if (existsSync(dir)) return dir
  }
  return resolve(parent, 'inline-core')
}

export function loadConfig(): ServerConfig {
  const env = process.env
  return {
    host: env.STORYLINE_HOST ?? '127.0.0.1',
    port: num(env.STORYLINE_PORT, 5173),
    coreUrl: env.INLINE_CORE_URL ?? 'http://127.0.0.1:8848',
    dataDir: env.STORYLINE_DATA_DIR ?? resolve(homedir(), '.storyline-server'),
    workspaceDir: env.STORYLINE_WORKSPACE_DIR ?? resolve(homedir(), 'StorylineProjects'),
    uiDir: env.STORYLINE_UI_DIR ?? resolve(process.cwd(), 'src/renderer/dist'),
    coreDir: env.INLINE_CORE_DIR ?? resolveCoreDir(),
    coreCmd: env.INLINE_CORE_CMD?.split(' ').filter(Boolean) ?? [
      'uv',
      'run',
      'python',
      '-m',
      'inline_core.server',
    ],
    coreManaged: env.STORYLINE_CORE_MANAGED !== 'false',
  }
}
