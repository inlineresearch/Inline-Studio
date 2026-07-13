/** IPC handlers for the fal DAG generation engine: run + cancel + resume, streaming events. */
import { BrowserWindow } from 'electron'
import { IpcChannels } from '@shared/ipc'
import type {
  GenerationProgressEvent,
  GenerationNodeDoneEvent,
  GenerationDoneEvent,
  GenerationErrorEvent,
} from '@shared/types'
import { handle } from './handler'
import {
  runGraph,
  cancelGeneration,
  resumePendingGenerations,
  type GenEmitter,
} from '../generation/executor'
import { runCoreWorkflow } from '../generation/coreExecutor'

/** Push an event to every renderer window (single-window app; mirrors the assets store). */
function broadcast(channel: string, payload: unknown): void {
  for (const w of BrowserWindow.getAllWindows()) w.webContents.send(channel, payload)
}

/** The main → renderer emitter, shared by fresh runs and restart-resume. */
function makeEmitter(): GenEmitter {
  return {
    progress: (frameId, fraction, status) =>
      broadcast(IpcChannels.events.generationProgress, {
        frameId,
        fraction,
        status,
      } satisfies GenerationProgressEvent),
    nodeDone: (frameId, take) => {
      broadcast(IpcChannels.events.generationNodeDone, {
        frameId,
        takeId: take.id,
      } satisfies GenerationNodeDoneEvent)
      // A new take exists on disk — refresh the Library "Generated Outputs" section.
      broadcast(IpcChannels.events.libraryChanged, undefined)
    },
    done: (targetFrameId) =>
      broadcast(IpcChannels.events.generationDone, {
        targetFrameId,
      } satisfies GenerationDoneEvent),
    error: (targetFrameId, error, frameId) =>
      broadcast(IpcChannels.events.generationError, {
        targetFrameId,
        frameId,
        error,
      } satisfies GenerationErrorEvent),
  }
}

export function registerGenerationHandlers(): void {
  handle<[string], void>(IpcChannels.generation.run, (frameId) => {
    if (typeof frameId !== 'string' || frameId.length === 0) throw new Error('Invalid frame id.')
    // Fire and forget: progress streams back via events; the invoke resolves immediately.
    void runGraph(frameId, makeEmitter())
  })

  handle<[string], void>(IpcChannels.generation.runWorkflow, (itemId) => {
    if (typeof itemId !== 'string' || itemId.length === 0) throw new Error('Invalid node id.')
    void runCoreWorkflow(itemId, makeEmitter())
  })

  handle<[string | undefined], void>(IpcChannels.generation.cancel, (frameId) => {
    void cancelGeneration(typeof frameId === 'string' && frameId.length > 0 ? frameId : undefined)
  })

  // Resume any generations that were still in flight when the app last closed.
  handle<[], void>(IpcChannels.generation.resumePending, () => {
    void resumePendingGenerations(makeEmitter())
  })
}
