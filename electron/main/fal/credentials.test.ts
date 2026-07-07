import { describe, it, expect, vi, beforeEach } from 'vitest'

// In-memory fake filesystem + a controllable safeStorage, so the credential store can be tested
// without Electron's runtime or touching disk.
const files = new Map<string, Buffer>()
let encryptionAvailable = true

vi.mock('electron', () => ({
  app: { getPath: () => '/userData' },
  safeStorage: {
    isEncryptionAvailable: () => encryptionAvailable,
    // A reversible stand-in for real OS encryption (prefix marker).
    encryptString: (s: string) => Buffer.concat([Buffer.from('ENC:'), Buffer.from(s, 'utf-8')]),
    decryptString: (b: Buffer) => b.subarray(4).toString('utf-8'),
  },
}))

vi.mock('node:fs', () => ({
  existsSync: (p: string) => files.has(p),
  readFileSync: (p: string) => {
    const b = files.get(p)
    if (!b) throw new Error('ENOENT')
    return b
  },
  writeFileSync: (p: string, data: Buffer) => {
    files.set(p, Buffer.isBuffer(data) ? data : Buffer.from(data))
  },
  rmSync: (p: string) => {
    files.delete(p)
  },
}))

import { getApiKey, setApiKey, clearApiKey, isConfigured } from './credentials'

beforeEach(() => {
  files.clear()
  encryptionAvailable = true
  delete process.env.FAL_KEY
})

describe('fal credentials', () => {
  it('round-trips an encrypted key', () => {
    expect(isConfigured()).toBe(false)
    setApiKey('  key-123  ')
    expect(isConfigured()).toBe(true)
    expect(getApiKey()).toBe('key-123') // trimmed
  })

  it('falls back to plaintext (with magic marker) when encryption is unavailable', () => {
    encryptionAvailable = false
    setApiKey('plain-key')
    // Stored file starts with the plaintext magic, not the ENC: marker.
    const stored = [...files.values()][0]
    expect(stored.toString('utf-8').startsWith('INLINESTUDIO_FAL_PLAINTEXT_V1')).toBe(true)
    expect(getApiKey()).toBe('plain-key')
  })

  it('reads the FAL_KEY env var when no key file exists', () => {
    process.env.FAL_KEY = 'env-key'
    expect(isConfigured()).toBe(false) // env doesn't count as "configured locally"
    expect(getApiKey()).toBe('env-key')
  })

  it('prefers a stored key over the env var', () => {
    process.env.FAL_KEY = 'env-key'
    setApiKey('stored-key')
    expect(getApiKey()).toBe('stored-key')
  })

  it('clears the stored key and falls back to env/null', () => {
    setApiKey('stored-key')
    clearApiKey()
    expect(isConfigured()).toBe(false)
    expect(getApiKey()).toBeNull()
  })

  it('rejects an empty key', () => {
    expect(() => setApiKey('   ')).toThrow(/empty/i)
  })
})
