/**
 * The native-capability seam: the few operations that differ by shell (dialogs, clipboard,
 * opening URLs, the OS keychain, the app data dir). Handlers call these instead of importing
 * `electron`, so the same handler modules run headless on the web server, which supplies its
 * own implementation. Dialog methods have no browser equivalent; the web shell rejects them and
 * the web client handles those flows itself (uploads, downloads).
 */

/** A file-dialog filter, e.g. { name: 'Media', extensions: ['png', 'mp4'] }. */
export interface FileFilter {
  name: string
  extensions: string[]
}

export interface DirectoryPickOptions {
  title?: string
  buttonLabel?: string
  createDirectory?: boolean
}

export interface FilePickOptions {
  title?: string
  buttonLabel?: string
  filters?: FileFilter[]
  multiple?: boolean
}

export interface SavePickOptions {
  title?: string
  defaultPath?: string
  filters?: FileFilter[]
}

export interface Capabilities {
  /** The running app version. */
  appVersion(): string
  /** App-global data dir for settings/recents/credentials/caches (userData, or the server data dir). */
  appDataDir(): string
  /** Open an http(s) URL outside the app (OS browser, or a new browser tab). */
  openExternal(url: string): Promise<void>
  /** Write text to the system clipboard. */
  writeClipboardText(text: string): void
  /** Copy an image file to the system clipboard. */
  copyImageFile(sourcePath: string): void
  /** Native directory picker; null if cancelled. */
  pickDirectory(opts?: DirectoryPickOptions): Promise<string | null>
  /** Native file picker (optionally multiple); [] if cancelled. */
  pickFiles(opts?: FilePickOptions): Promise<string[]>
  /** Native save-location picker; null if cancelled. */
  pickSavePath(opts?: SavePickOptions): Promise<string | null>
  /** Whether the OS provides real secret encryption. */
  isEncryptionAvailable(): boolean
  /** Encrypt a secret for at-rest storage (only when isEncryptionAvailable()). */
  encryptSecret(plain: string): Buffer
  /** Decrypt a secret produced by encryptSecret. */
  decryptSecret(data: Buffer): string
}
