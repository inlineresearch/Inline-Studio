/**
 * Copy text to the clipboard, resiliently.
 *
 * The async Clipboard API (`navigator.clipboard`) only exists in a *secure context* - HTTPS or
 * `localhost`. When Core is reached over a plain-HTTP LAN address (`webui.sh --listen`, e.g.
 * `http://192.168.1.5:8848`), `navigator.clipboard` is `undefined`, so calling `.writeText` on it
 * throws synchronously and any copy button appears dead. Fall back to the legacy
 * `document.execCommand('copy')` over a hidden textarea, which works in non-secure contexts.
 *
 * Returns whether the copy succeeded.
 */
export async function copyText(text: string): Promise<boolean> {
  // Preferred path: the async Clipboard API, when available in a secure context.
  if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text)
      return true
    } catch {
      // Permission denied or blocked - fall through to the legacy path.
    }
  }
  return legacyCopy(text)
}

function legacyCopy(text: string): boolean {
  if (typeof document === 'undefined') return false
  const ta = document.createElement('textarea')
  ta.value = text
  // Keep it out of view and unfocusable-scrolling, but still selectable.
  ta.setAttribute('readonly', '')
  ta.style.position = 'fixed'
  ta.style.top = '-9999px'
  ta.style.opacity = '0'
  document.body.appendChild(ta)
  try {
    ta.select()
    ta.setSelectionRange(0, text.length)
    return document.execCommand('copy')
  } catch {
    return false
  } finally {
    document.body.removeChild(ta)
  }
}
