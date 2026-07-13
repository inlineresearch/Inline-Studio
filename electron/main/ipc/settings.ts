/** IPC handlers for app-global settings. */
import { IpcChannels } from '@shared/ipc'
import type { AppSettings } from '@shared/types'
import { handle } from './handler'
import { getSettings, setComfyUrl, setCoreUrl } from '../settings/store'

export function registerSettingsHandlers(): void {
  handle<[], AppSettings>(IpcChannels.settings.get, () => getSettings())
  handle<[string], AppSettings>(IpcChannels.settings.setComfyUrl, (url) => {
    if (typeof url !== 'string') throw new Error('Invalid URL.')
    return setComfyUrl(url)
  })
  handle<[string], AppSettings>(IpcChannels.settings.setCoreUrl, (url) => {
    if (typeof url !== 'string') throw new Error('Invalid URL.')
    return setCoreUrl(url)
  })
}
