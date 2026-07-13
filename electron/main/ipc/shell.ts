/** IPC for opening external URLs in the user's default browser. */
import { IpcChannels } from '@shared/ipc'
import { handle } from './handler'
import { caps } from '../capabilities'

export function registerShellHandlers(): void {
  handle<[string], void>(IpcChannels.shell.openExternal, async (url) => {
    if (typeof url !== 'string' || !/^https?:\/\//i.test(url)) {
      throw new Error('Only http(s) URLs can be opened.')
    }
    await caps().openExternal(url)
  })
}
