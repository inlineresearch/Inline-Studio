/** The Electron transport: binds each handler to ipcMain, dropping the event arg. */
import { ipcMain } from 'electron'
import type { IpcMainInvokeEvent } from 'electron'
import type { Invoker, Transport } from './handler'

export const electronTransport: Transport = {
  register(channel: string, invoke: Invoker): void {
    ipcMain.handle(channel, (_event: IpcMainInvokeEvent, ...args: unknown[]) => invoke(...args))
  },
}
