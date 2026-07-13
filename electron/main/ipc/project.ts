/**
 * IPC handlers for project lifecycle. Each validates its payload (untrusted —
 * it comes from the renderer) before touching the filesystem.
 */
import { join } from 'node:path'
import { IpcChannels, type CreateProjectInput } from '@shared/ipc'
import type { Project, RecentProject, ProjectMediaDirs, ProjectExportResult } from '@shared/types'
import { handle } from './handler'
import { caps } from '../capabilities'
import { createProject, openProject, getCurrentProject, isProjectFolder } from '../project/store'
import { extractProjectZip } from '../project/import'
import { exportProject } from '../export/project'
import { listRecents } from '../project/recents'
import { getOpenProjectFolder } from '../db'

function assertCreateInput(input: unknown): asserts input is CreateProjectInput {
  if (
    typeof input !== 'object' ||
    input === null ||
    typeof (input as CreateProjectInput).name !== 'string' ||
    typeof (input as CreateProjectInput).parentDir !== 'string'
  ) {
    throw new Error('Invalid create-project input.')
  }
  if ((input as CreateProjectInput).name.trim().length === 0) {
    throw new Error('Project name is required.')
  }
}

export function registerProjectHandlers(): void {
  handle<[CreateProjectInput], Project>(IpcChannels.project.create, (input) => {
    assertCreateInput(input)
    return createProject(input)
  })

  handle<[string], Project>(IpcChannels.project.open, (path) => {
    if (typeof path !== 'string' || path.length === 0) throw new Error('Invalid project path.')
    return openProject(path)
  })

  handle<[], Project | null>(IpcChannels.project.openDialog, async () => {
    const folder = await caps().pickDirectory({
      title: 'Open Inline Studio Project',
      buttonLabel: 'Open Project',
    })
    if (!folder) return null
    if (!isProjectFolder(folder)) {
      throw new Error('That folder is not a Inline Studio project.')
    }
    return openProject(folder)
  })

  handle<[], Project | null>(IpcChannels.project.openZip, async () => {
    const [zip] = await caps().pickFiles({
      title: 'Open Inline Studio Project (.zip)',
      filters: [{ name: 'Inline Studio project', extensions: ['zip'] }],
      buttonLabel: 'Open Project',
    })
    if (!zip) return null
    const dest = await extractProjectZip(zip)
    if (!isProjectFolder(dest)) {
      throw new Error('That .zip does not contain an Inline Studio project.')
    }
    return openProject(dest)
  })

  handle<[], RecentProject[]>(IpcChannels.project.listRecent, () => listRecents())

  handle<[], Project | null>(IpcChannels.project.current, () => getCurrentProject())

  handle<[], ProjectMediaDirs>(IpcChannels.project.mediaDirs, () => {
    const folder = getOpenProjectFolder()
    if (!folder) throw new Error('No project is open.')
    return { inputDir: join(folder, 'assets'), outputDir: join(folder, 'takes') }
  })

  handle<[string], ProjectExportResult | null>(IpcChannels.project.export, (path) => {
    if (typeof path !== 'string' || path.length === 0) throw new Error('Invalid project path.')
    return exportProject(path)
  })

  handle<[], string | null>(IpcChannels.dialog.pickDirectory, () =>
    caps().pickDirectory({ title: 'Choose a location', createDirectory: true }),
  )
}
