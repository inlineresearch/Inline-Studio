/**
 * Gate workflow actions (Link / Open Workflow) on ComfyUI being reachable. Pings the
 * backend at call time (so it doesn't depend on a running status poll). When ComfyUI
 * isn't connected it alerts the user in plain language and runs `onDisconnected` — used
 * to send them to the Generate tab where they can connect — and returns false so the
 * caller can bail before attempting a link/upload that would fail.
 */
import { studio } from '@/lib/studio'

export async function requireComfyConnected(onDisconnected?: () => void): Promise<boolean> {
  let connected = false
  try {
    const res = await studio().comfy.status()
    connected = res.ok && res.value.running
  } catch {
    connected = false
  }
  if (connected) return true

  window.alert(
    'ComfyUI isn’t connected yet.\n\nConnect it on the Generate tab, then link or open your workflow.',
  )
  onDisconnected?.()
  return false
}
