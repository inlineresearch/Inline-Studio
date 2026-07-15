/**
 * Keeps inline-core (the Python generation engine) available so the user runs one command. If a core
 * is already reachable at the configured URL, we use it. Otherwise, when managed, we spawn it from
 * its checkout and poll until healthy. Failures degrade gracefully: the rest of Storyline works
 * without core, and only generation is affected until core is up. The core URL comes from settings
 * (getSettings), the same source the generation handlers use, so they always agree.
 */
import { spawn, type ChildProcess } from 'node:child_process'
import { homedir } from 'node:os'
import { resolve } from 'node:path'
import { getSettings } from '@main/settings/store'
import type { ServerConfig } from '../config'

function coreUrl(): string {
  return getSettings().coreUrl.replace(/\/+$/, '')
}

/**
 * Ensure `~/.local/bin` is on PATH for the spawned core. `uv` installs there, but a non-login/
 * non-interactive shell (e.g. the one that launches this server from an IDE) may not have sourced
 * the profile that adds it — leaving `spawn uv` to fail with ENOENT. Prepend it defensively.
 */
function pathWithLocalBin(): string {
  const localBin = resolve(homedir(), '.local', 'bin')
  const current = process.env.PATH ?? ''
  return current.split(':').includes(localBin) ? current : `${localBin}:${current}`
}

async function isCoreUp(): Promise<boolean> {
  try {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 1500)
    const res = await fetch(`${coreUrl()}/v1/health`, { signal: controller.signal })
    clearTimeout(timer)
    return res.ok
  } catch {
    return false
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

async function waitForCore(timeoutMs = 40000): Promise<boolean> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (await isCoreUp()) return true
    await sleep(1000)
  }
  return false
}

/** Ensure inline-core is running; returns the spawned child (to shut down) or null. */
export async function startCore(config: ServerConfig): Promise<ChildProcess | null> {
  if (await isCoreUp()) {
    console.log(`Connected to Inline Core at ${coreUrl()}`)
    return null
  }
  if (!config.coreManaged) {
    console.warn(`Inline Core not reachable at ${coreUrl()}; start it or generation will fail.`)
    return null
  }

  const [cmd, ...args] = config.coreCmd
  const { hostname, port } = new URL(coreUrl())
  console.log(`Starting Inline Core: ${config.coreCmd.join(' ')} (in ${config.coreDir})`)
  let child: ChildProcess
  try {
    child = spawn(cmd, args, {
      cwd: config.coreDir,
      stdio: 'inherit',
      env: {
        ...process.env,
        PATH: pathWithLocalBin(),
        INLINE_HOST: hostname,
        INLINE_PORT: port || '8848',
      },
    })
  } catch (e) {
    console.warn(
      `Could not start Inline Core: ${errMsg(e)}. Start it manually or set INLINE_CORE_CMD.`,
    )
    return null
  }
  child.on('error', (e) => {
    console.warn(
      `Inline Core failed to launch: ${e.message}. Start it manually or set INLINE_CORE_CMD.`,
    )
  })
  void waitForCore().then((up) => {
    console.log(up ? 'Inline Core is ready.' : 'Inline Core did not become ready in time.')
  })
  return child
}

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}
