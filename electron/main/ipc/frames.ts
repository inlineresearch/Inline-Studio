/** IPC handlers for frames + their takes. */
import { IpcChannels } from '@shared/ipc'
import type { Frame, Take, FrameInput } from '@shared/types'
import { handle } from './handler'
import {
  listFrames,
  importAsFrames,
  addFromAsset,
  renameFrame,
  reorderFrames,
  deleteFrame,
  cloneFrame,
  unlinkWorkflow,
  setHero,
  listTakes,
  heroTakes,
  listInputs,
  addInput,
  addSourceInput,
  removeInput,
  reorderInputs,
  listAllTakes,
  deleteTake,
  setFalParams,
  setModel,
} from '../frames/store'

function str(v: unknown, label: string): string {
  if (typeof v !== 'string' || v.length === 0) throw new Error(`Invalid ${label}.`)
  return v
}

export function registerFrameHandlers(): void {
  handle<[], Frame[]>(IpcChannels.frames.list, () => listFrames())
  handle<[], Frame[]>(IpcChannels.frames.importAsFrames, () => importAsFrames())
  handle<[string], Frame>(IpcChannels.frames.addFromAsset, (assetId) =>
    addFromAsset(str(assetId, 'asset id')),
  )
  handle<[string, string], Frame>(IpcChannels.frames.rename, (id, name) => {
    if (typeof name !== 'string') throw new Error('Invalid name.')
    return renameFrame(str(id, 'frame id'), name)
  })
  handle<[string[]], void>(IpcChannels.frames.reorder, (orderedIds) => {
    if (!Array.isArray(orderedIds) || orderedIds.some((x) => typeof x !== 'string')) {
      throw new Error('Invalid ordering.')
    }
    reorderFrames(orderedIds)
  })
  handle<[string], void>(IpcChannels.frames.delete, (id) => deleteFrame(str(id, 'frame id')))
  handle<[string], Frame>(IpcChannels.frames.clone, (id) => cloneFrame(str(id, 'frame id')))
  handle<[string], Frame>(IpcChannels.frames.unlink, (id) => unlinkWorkflow(str(id, 'frame id')))
  handle<[string, string | null], Frame>(IpcChannels.frames.setHero, (id, takeId) => {
    if (takeId !== null && typeof takeId !== 'string') throw new Error('Invalid take id.')
    return setHero(str(id, 'frame id'), takeId)
  })
  handle<[string], Take[]>(IpcChannels.frames.listTakes, (frameId) =>
    listTakes(str(frameId, 'frame id')),
  )
  handle<[], Take[]>(IpcChannels.frames.heroTakes, () => heroTakes())
  handle<[], FrameInput[]>(IpcChannels.frames.listInputs, () => listInputs())
  handle<[string, string], FrameInput>(IpcChannels.frames.addInput, (frameId, assetId) =>
    addInput(str(frameId, 'frame id'), str(assetId, 'asset id')),
  )
  handle<[string, string], FrameInput>(
    IpcChannels.frames.addSourceInput,
    (frameId, sourceFrameId) =>
      addSourceInput(str(frameId, 'frame id'), str(sourceFrameId, 'source frame id')),
  )
  handle<[string, string], void>(IpcChannels.frames.removeInput, (frameId, assetId) =>
    removeInput(str(frameId, 'frame id'), str(assetId, 'asset id')),
  )
  handle<[string, string[]], void>(IpcChannels.frames.reorderInputs, (frameId, orderedAssetIds) => {
    if (!Array.isArray(orderedAssetIds) || orderedAssetIds.some((x) => typeof x !== 'string')) {
      throw new Error('Invalid input ordering.')
    }
    reorderInputs(str(frameId, 'frame id'), orderedAssetIds)
  })
  handle<[], Take[]>(IpcChannels.frames.listAllTakes, () => listAllTakes())
  handle<[string], void>(IpcChannels.frames.deleteTake, (takeId) =>
    deleteTake(str(takeId, 'take id')),
  )
  handle<[string, Record<string, unknown>], Frame>(
    IpcChannels.frames.setFalParams,
    (frameId, params) => {
      if (typeof params !== 'object' || params === null) throw new Error('Invalid params.')
      return setFalParams(str(frameId, 'frame id'), params)
    },
  )
  handle<[string, string], Frame>(IpcChannels.frames.setModel, (frameId, modelId) =>
    setModel(str(frameId, 'frame id'), str(modelId, 'model id')),
  )
}
