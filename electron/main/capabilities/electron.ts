/** The Electron implementation of the native-capability seam. */
import { app, dialog, clipboard, nativeImage, shell, safeStorage } from 'electron'
import type { Capabilities } from './types'

export const electronCapabilities: Capabilities = {
  appVersion: () => app.getVersion(),
  appDataDir: () => app.getPath('userData'),

  async openExternal(url) {
    await shell.openExternal(url)
  },

  writeClipboardText(text) {
    clipboard.writeText(text)
  },

  copyImageFile(sourcePath) {
    const image = nativeImage.createFromPath(sourcePath)
    if (image.isEmpty()) throw new Error('That file could not be read as an image.')
    clipboard.writeImage(image)
  },

  async pickDirectory(opts) {
    const properties: Array<'openDirectory' | 'createDirectory'> = ['openDirectory']
    if (opts?.createDirectory) properties.push('createDirectory')
    const r = await dialog.showOpenDialog({
      title: opts?.title,
      buttonLabel: opts?.buttonLabel,
      properties,
    })
    return r.canceled || r.filePaths.length === 0 ? null : r.filePaths[0]
  },

  async pickFiles(opts) {
    const properties: Array<'openFile' | 'multiSelections'> = ['openFile']
    if (opts?.multiple) properties.push('multiSelections')
    const r = await dialog.showOpenDialog({
      title: opts?.title,
      buttonLabel: opts?.buttonLabel,
      filters: opts?.filters,
      properties,
    })
    return r.canceled ? [] : r.filePaths
  },

  async pickSavePath(opts) {
    const r = await dialog.showSaveDialog({
      title: opts?.title,
      defaultPath: opts?.defaultPath,
      filters: opts?.filters,
    })
    return r.canceled || !r.filePath ? null : r.filePath
  },

  isEncryptionAvailable: () => safeStorage.isEncryptionAvailable(),
  encryptSecret: (plain) => safeStorage.encryptString(plain),
  decryptSecret: (data) => safeStorage.decryptString(data),
}
