/**
 * Browser-shell capabilities. Native dialogs, clipboard, and opening links have no server-side
 * equivalent, so those flows are handled by the web client (uploads, downloads, browser APIs) and
 * these reject clearly if ever reached over RPC. Only appVersion, appDataDir, and the (absent)
 * encryption apply server-side.
 */
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import type { Capabilities } from '@main/capabilities/types'

export function webCapabilities(dataDir: string, workspaceDir: string): Capabilities {
  const unsupported = (what: string): never => {
    throw new Error(`${what} is not available in the browser session.`)
  }
  return {
    appVersion: () => readVersion(),
    appDataDir: () => dataDir,
    openExternal: () => unsupported('Opening an external link'),
    writeClipboardText: () => unsupported('Clipboard write'),
    copyImageFile: () => unsupported('Copying an image'),
    // No native picker in a browser: new projects go into the server's workspace dir.
    pickDirectory: () => Promise.resolve(workspaceDir),
    pickFiles: () => unsupported('A file picker'),
    pickSavePath: () => unsupported('A save dialog'),
    isEncryptionAvailable: () => false,
    encryptSecret: () => unsupported('Secret encryption'),
    decryptSecret: () => unsupported('Secret decryption'),
  }
}

function readVersion(): string {
  try {
    const raw = readFileSync(resolve(process.cwd(), 'package.json'), 'utf-8')
    return (JSON.parse(raw) as { version?: string }).version ?? '0.0.0'
  } catch {
    return '0.0.0'
  }
}
