/**
 * Encrypted storage for the user's fal.ai API key — the secret the fal generation engine needs.
 * Encrypted with Electron `safeStorage` (OS keychain-backed) in its own file under userData;
 * falls back to a clearly-marked plaintext file (0600) on platforms without a secure keystore.
 * The key lives only in the main process, never crossing the bridge.
 *
 * `getApiKey()` also falls back to the `FAL_KEY` env var (see `.env.example`) so headless/dev
 * runs can supply a key without the UI, matching the settings-store env-fallback pattern.
 */
import { app, safeStorage } from 'electron'
import { join } from 'node:path'
import { readFileSync, writeFileSync, existsSync, rmSync } from 'node:fs'

/** Prefix marking an unencrypted fallback file, so reads know how to decode it. */
const PLAINTEXT_MAGIC = Buffer.from('INLINESTUDIO_FAL_PLAINTEXT_V1\n', 'utf-8')

function keyFile(): string {
  return join(app.getPath('userData'), 'fal-credentials.bin')
}

/** Whether the OS provides real encryption (false on some headless Linux). */
export function isEncryptionAvailable(): boolean {
  return safeStorage.isEncryptionAvailable()
}

/** True if a key is saved locally (ignores the env fallback). */
export function isConfigured(): boolean {
  return existsSync(keyFile())
}

/** Persist the fal API key, encrypted when the OS supports it. */
export function setApiKey(key: string): void {
  const trimmed = key.trim()
  if (!trimmed) throw new Error('API key is empty.')
  if (safeStorage.isEncryptionAvailable()) {
    writeFileSync(keyFile(), safeStorage.encryptString(trimmed))
  } else {
    writeFileSync(keyFile(), Buffer.concat([PLAINTEXT_MAGIC, Buffer.from(trimmed, 'utf-8')]), {
      mode: 0o600,
    })
  }
}

/** Decrypt and return the stored key, else the `FAL_KEY` env var, else null. */
export function getApiKey(): string | null {
  try {
    if (existsSync(keyFile())) {
      const buf = readFileSync(keyFile())
      if (buf.subarray(0, PLAINTEXT_MAGIC.length).equals(PLAINTEXT_MAGIC)) {
        return buf.subarray(PLAINTEXT_MAGIC.length).toString('utf-8')
      }
      return safeStorage.decryptString(buf)
    }
  } catch {
    // fall through to env
  }
  const fromEnv = process.env.FAL_KEY?.trim()
  return fromEnv && fromEnv.length > 0 ? fromEnv : null
}

export function clearApiKey(): void {
  rmSync(keyFile(), { force: true })
}
