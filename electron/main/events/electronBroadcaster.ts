/** The Electron broadcaster: sends to every renderer window (single-window app in practice). */
import { BrowserWindow } from 'electron'
import type { Broadcaster } from './broadcaster'

export const electronBroadcaster: Broadcaster = {
  send(channel: string, payload?: unknown): void {
    for (const w of BrowserWindow.getAllWindows()) w.webContents.send(channel, payload)
  },
}
