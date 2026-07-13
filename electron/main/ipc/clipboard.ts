/** IPC handler for writing text to the system clipboard. */
import { IpcChannels } from '@shared/ipc'
import { handle } from './handler'
import { caps } from '../capabilities'

export function registerClipboardHandlers(): void {
  handle<[string], void>(IpcChannels.clipboard.writeText, (text) => {
    if (typeof text !== 'string') throw new Error('Invalid clipboard text.')
    caps().writeClipboardText(text)
  })
}
