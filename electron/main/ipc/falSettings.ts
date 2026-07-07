/** IPC handlers for the fal.ai API key (bring-your-own-key). */
import { IpcChannels } from '@shared/ipc'
import type { ApiKeyStatus } from '@shared/types'
import { handle } from './handler'
import { isConfigured, isEncryptionAvailable, setApiKey, clearApiKey } from '../fal/credentials'

function getStatus(): ApiKeyStatus {
  return { configured: isConfigured(), encrypted: isEncryptionAvailable() }
}

export function registerFalSettingsHandlers(): void {
  handle<[], ApiKeyStatus>(IpcChannels.falSettings.status, () => getStatus())
  handle<[string], ApiKeyStatus>(IpcChannels.falSettings.setApiKey, (key) => {
    if (typeof key !== 'string') throw new Error('Invalid API key.')
    // We store without a remote validation call — fal has no cheap probe, and a real check would
    // cost a generation. A bad key surfaces as a clear error on the first run instead.
    setApiKey(key)
    return getStatus()
  })
  handle<[], ApiKeyStatus>(IpcChannels.falSettings.clearApiKey, () => {
    clearApiKey()
    return getStatus()
  })
}
