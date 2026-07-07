/** IPC handlers for the ComfyUI bridge. */
import { IpcChannels } from '@shared/ipc'
import type { Take, ComfyStatus, Frame, ComfyRun, ComfyOutput } from '@shared/types'
import { handle } from './handler'
import {
  ping,
  linkFrameWorkflow,
  uploadFrameInputs,
  pullWorkflowToProject,
  saveLiveWorkflow,
  pushWorkflowFromProject,
  pullLatestToFrame,
  latestRun,
  captureOutput,
} from '../comfy/client'

function str(v: unknown, label: string): string {
  if (typeof v !== 'string' || v.length === 0) throw new Error(`Invalid ${label}.`)
  return v
}

function asOutput(v: unknown): ComfyOutput {
  if (typeof v !== 'object' || v === null || typeof (v as ComfyOutput).filename !== 'string') {
    throw new Error('Invalid output.')
  }
  return v as ComfyOutput
}

/** A serialized workflow graph from the renderer must at least be a plain object. */
function asWorkflow(v: unknown): Record<string, unknown> {
  if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new Error('Invalid workflow.')
  return v as Record<string, unknown>
}

export function registerComfyHandlers(): void {
  handle<[], ComfyStatus>(IpcChannels.comfy.status, () => ping())
  handle<[string], Frame>(IpcChannels.comfy.linkFrame, (frameId) =>
    linkFrameWorkflow(str(frameId, 'frame id')),
  )
  handle<[string], string[]>(IpcChannels.comfy.uploadInputs, (frameId) =>
    uploadFrameInputs(str(frameId, 'frame id')),
  )
  handle<[string], boolean>(IpcChannels.comfy.pullWorkflow, (frameId) =>
    pullWorkflowToProject(str(frameId, 'frame id')),
  )
  handle<[string, unknown], Frame | null>(IpcChannels.comfy.saveLiveWorkflow, (frameId, workflow) =>
    saveLiveWorkflow(str(frameId, 'frame id'), asWorkflow(workflow)),
  )
  handle<[string], void>(IpcChannels.comfy.pushWorkflow, (frameId) =>
    pushWorkflowFromProject(str(frameId, 'frame id')),
  )
  handle<[string], Take>(IpcChannels.comfy.pullLatest, (frameId) =>
    pullLatestToFrame(str(frameId, 'frame id')),
  )
  handle<[], ComfyRun | null>(IpcChannels.comfy.latestRun, () => latestRun())
  handle<[string, ComfyOutput], Take>(IpcChannels.comfy.captureOutput, (frameId, output) =>
    captureOutput(str(frameId, 'frame id'), asOutput(output)),
  )
}
