/** IPC handlers for the moodboard. Payloads come from the renderer — validate them. */
import { IpcChannels, type MoodboardItemPatch } from '@shared/ipc'
import type { MoodboardItem, MoodboardConnector, MoodboardSnapshot } from '@shared/types'
import { handle } from './handler'
import {
  listBoard,
  addAssetItem,
  addTextItem,
  addFrameFromAsset,
  addEmptyFrame,
  addFrameItem,
  addPreview,
  addLayer,
  addDirector,
  addTrim,
  addGenNode,
  addPrompt,
  updateItem,
  deleteItem,
  importAndPlace,
  createConnector,
  deleteConnector,
  setConnectorVolume,
  replaceBoard,
} from '../moodboard/store'

function num(v: unknown, label: string): number {
  if (typeof v !== 'number' || Number.isNaN(v)) throw new Error(`Invalid ${label}.`)
  return v
}

function str(v: unknown, label: string): string {
  if (typeof v !== 'string' || v.length === 0) throw new Error(`Invalid ${label}.`)
  return v
}

export function registerMoodboardHandlers(): void {
  handle<[], MoodboardSnapshot>(IpcChannels.moodboard.list, () => listBoard())

  handle<[string, number, number], MoodboardItem>(IpcChannels.moodboard.addAsset, (assetId, x, y) =>
    addAssetItem(str(assetId, 'asset id'), num(x, 'x'), num(y, 'y')),
  )

  handle<[number, number], MoodboardItem>(IpcChannels.moodboard.addText, (x, y) =>
    addTextItem(num(x, 'x'), num(y, 'y')),
  )

  handle<[string, number, number], MoodboardItem>(
    IpcChannels.moodboard.addFrameFromAsset,
    (assetId, x, y) => addFrameFromAsset(str(assetId, 'asset id'), num(x, 'x'), num(y, 'y')),
  )

  handle<[number, number], MoodboardItem>(IpcChannels.moodboard.addEmptyFrame, (x, y) =>
    addEmptyFrame(num(x, 'x'), num(y, 'y')),
  )

  handle<[string, number, number], MoodboardItem>(
    IpcChannels.moodboard.addFrameItem,
    (frameId, x, y) => addFrameItem(str(frameId, 'frame id'), num(x, 'x'), num(y, 'y')),
  )

  handle<[number, number], MoodboardItem>(IpcChannels.moodboard.addPreview, (x, y) =>
    addPreview(num(x, 'x'), num(y, 'y')),
  )

  handle<[number, number], MoodboardItem>(IpcChannels.moodboard.addLayer, (x, y) =>
    addLayer(num(x, 'x'), num(y, 'y')),
  )

  handle<[number, number], MoodboardItem>(IpcChannels.moodboard.addDirector, (x, y) =>
    addDirector(num(x, 'x'), num(y, 'y')),
  )

  handle<[number, number], MoodboardItem>(IpcChannels.moodboard.addTrim, (x, y) =>
    addTrim(num(x, 'x'), num(y, 'y')),
  )

  handle<[string, number, number], MoodboardItem>(
    IpcChannels.moodboard.addGenNode,
    (modelId, x, y) => addGenNode(str(modelId, 'model id'), num(x, 'x'), num(y, 'y')),
  )

  handle<[number, number], MoodboardItem>(IpcChannels.moodboard.addPrompt, (x, y) =>
    addPrompt(num(x, 'x'), num(y, 'y')),
  )

  handle<[string, MoodboardItemPatch], MoodboardItem>(
    IpcChannels.moodboard.updateItem,
    (id, patch) => {
      if (typeof patch !== 'object' || patch === null) throw new Error('Invalid patch.')
      return updateItem(str(id, 'item id'), patch)
    },
  )

  handle<[string], void>(IpcChannels.moodboard.deleteItem, (id) => deleteItem(str(id, 'item id')))

  handle<[number, number], MoodboardItem[]>(IpcChannels.moodboard.importAndPlace, (x, y) =>
    importAndPlace(num(x, 'x'), num(y, 'y')),
  )

  handle<[string, string, string | null, string | null], MoodboardConnector>(
    IpcChannels.moodboard.createConnector,
    (from, to, sourceHandle, targetHandle) =>
      createConnector(
        str(from, 'from id'),
        str(to, 'to id'),
        typeof sourceHandle === 'string' ? sourceHandle : null,
        typeof targetHandle === 'string' ? targetHandle : null,
      ),
  )

  handle<[string], void>(IpcChannels.moodboard.deleteConnector, (id) =>
    deleteConnector(str(id, 'connector id')),
  )

  handle<[string, number], void>(IpcChannels.moodboard.setConnectorVolume, (id, volume) =>
    setConnectorVolume(str(id, 'connector id'), num(volume, 'volume')),
  )

  handle<[MoodboardItem[], MoodboardConnector[]], void>(
    IpcChannels.moodboard.replaceBoard,
    (items, connectors) => {
      if (!Array.isArray(items) || !Array.isArray(connectors))
        throw new Error('Invalid board snapshot.')
      replaceBoard(items, connectors)
    },
  )
}
