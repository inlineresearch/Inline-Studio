/**
 * Electron main entry. Owns the app window with a strict security baseline
 * (see CLAUDE.md): contextIsolation on, nodeIntegration off, sandbox on. The
 * renderer reaches the outside world only through the preload bridge + IPC.
 */
import { app, BrowserWindow, shell, nativeImage } from 'electron'
import { join } from 'node:path'
import { existsSync } from 'node:fs'
import { registerIpcHandlers } from './ipc'
import { registerUpdateHandlers } from './ipc/updates'
import { setTransport } from './ipc/handler'
import { electronTransport } from './ipc/electronTransport'
import { setBroadcaster } from './events/broadcaster'
import { electronBroadcaster } from './events/electronBroadcaster'
import { setCapabilities } from './capabilities'
import { electronCapabilities } from './capabilities/electron'
import { registerMediaScheme, registerMediaProtocol } from './media/protocol'
import { closeProjectDb } from './db'
import { initAutoUpdater } from './updater'

const isDev = !app.isPackaged

// Must be registered before the app is ready.
registerMediaScheme()

/**
 * In dev, macOS takes the dock icon from the (default Electron) bundle, so our SL
 * icon never shows. Point the dock at the build icon. No-op once packaged — the
 * .app already carries `build/icon.icns`.
 */
function setDevDockIcon(): void {
  if (!isDev || process.platform !== 'darwin' || !app.dock) return
  // __dirname is <project>/out/main in dev; the icon lives at <project>/build.
  const iconPath = join(__dirname, '../../build/icon.png')
  if (!existsSync(iconPath)) return
  const img = nativeImage.createFromPath(iconPath)
  if (!img.isEmpty()) app.dock.setIcon(img)
}

function createMainWindow(): BrowserWindow {
  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1024,
    minHeight: 640,
    show: false,
    backgroundColor: '#16171b',
    title: 'Inline Studio',
    webPreferences: {
      preload: join(__dirname, '../preload/index.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      // Enables the <webview> used to embed + drive the user's local ComfyUI
      // (Generate tab). We only ever load the user-configured Comfy URL in it.
      webviewTag: true,
    },
  })

  win.once('ready-to-show', () => win.show())

  // Open any external links (e.g. the ComfyUI escape hatch) in the OS browser,
  // never inside the app window.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('http://') || url.startsWith('https://')) {
      void shell.openExternal(url)
    }
    return { action: 'deny' }
  })

  const devUrl = process.env.ELECTRON_RENDERER_URL
  if (isDev && devUrl) {
    void win.loadURL(devUrl)
    win.webContents.openDevTools({ mode: 'detach' })
  } else {
    void win.loadFile(join(__dirname, '../renderer/index.html'))
  }

  return win
}

app.whenReady().then(() => {
  registerMediaProtocol()
  // Bind the Electron shell seams before registering handlers, then add the desktop-only
  // auto-update handlers (the shared list stays runnable on the headless web server).
  setTransport(electronTransport)
  setBroadcaster(electronBroadcaster)
  setCapabilities(electronCapabilities)
  registerIpcHandlers()
  registerUpdateHandlers()
  setDevDockIcon()
  createMainWindow()
  // Window exists now, so it can receive update broadcasts.
  initAutoUpdater()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createMainWindow()
  })
})

app.on('window-all-closed', () => {
  closeProjectDb()
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', () => {
  closeProjectDb()
})
