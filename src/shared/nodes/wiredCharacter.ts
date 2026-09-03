/** Mirrors `studio/fal.py::wired_character`, for pricing only - the run still resolves it in Core. */
import type { MoodboardConnector, MoodboardItem } from '../types'

/** Where each node that can name an already-saved character keeps it. */
const CHARACTER_FILE_PARAM: Record<string, string> = {
  'character/load': 'file',
  'character/write': 'filename',
}

/** The `.char` a node's Character port is fed by, or null. */
export function wiredCharacterFile(
  itemId: string,
  connectors: MoodboardConnector[],
  items: MoodboardItem[],
): string | null {
  for (const c of connectors) {
    if (c.toItemId !== itemId) continue
    if ((c.data?.targetHandle as string | undefined) !== 'character') continue
    const core = items.find((it) => it.id === c.fromItemId)?.data?.core
    const key = core ? CHARACTER_FILE_PARAM[core.type] : undefined
    const raw = key ? core?.params?.[key] : undefined
    // A path reduces to its last part and gains the suffix, the way `library.target_name` does.
    const name = String(raw ?? '')
      .trim()
      .replace(/\\/g, '/')
      .split('/')
      .pop()
      ?.trim()
    if (name) return name.toLowerCase().endsWith('.char') ? name : `${name}.char`
  }
  return null
}
