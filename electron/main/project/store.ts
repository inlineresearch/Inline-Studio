/**
 * Project lifecycle: create/open a `.inlinestudio` folder and load its DB.
 *
 * A project on disk is a portable folder:
 *   MyFilm.inlinestudio/
 *     project.db   assets/   takes/   thumbs/
 */
import { join } from 'node:path'
import { mkdirSync, existsSync, readdirSync } from 'node:fs'
import { randomUUID } from 'node:crypto'
import type { Project } from '@shared/types'
import { openProjectDb, getDb } from '../db'
import { recordRecent } from './recents'
import { backfillVideoAssets, backfillAudioAssets, backfillImageAssets } from '../assets/store'
import { createEmptyFrame } from '../frames/store'
import { addFrameItem } from '../moodboard/store'

/** Extension for newly-created projects. */
const PROJECT_EXT = '.inlinestudio'
/** Also openable for backward compatibility (projects from when the app was "Storyline"). */
const LEGACY_PROJECT_EXTS = ['.storyline']
/** All folder extensions recognised as a project (new + legacy). */
const PROJECT_EXTS = [PROJECT_EXT, ...LEGACY_PROJECT_EXTS]

const isProjectExt = (folder: string): boolean => PROJECT_EXTS.some((ext) => folder.endsWith(ext))

const SUBDIRS = ['assets', 'takes', 'thumbs']

let currentProject: Project | null = null

function sanitizeFolderName(name: string): string {
  const base = name
    .trim()
    .replace(/[^\w.-]+/g, '-')
    .replace(/^-+|-+$/g, '')
  return base.length > 0 ? base : 'untitled'
}

interface ProjectRow {
  id: string
  name: string
  created_at: number
  updated_at: number
}

function loadProjectRow(folder: string): Project {
  const db = getDb()
  const row = db.prepare('SELECT id, name, created_at, updated_at FROM project LIMIT 1').get() as
    | ProjectRow
    | undefined
  if (!row) throw new Error('project.db is missing its project record.')
  return {
    id: row.id,
    name: row.name,
    path: folder,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  }
}

export function createProject(input: { name: string; parentDir: string }): Project {
  const folderName = `${sanitizeFolderName(input.name)}${PROJECT_EXT}`
  const folder = join(input.parentDir, folderName)
  if (existsSync(folder)) {
    throw new Error(`A project already exists at ${folder}`)
  }
  mkdirSync(folder, { recursive: true })
  for (const sub of SUBDIRS) mkdirSync(join(folder, sub), { recursive: true })

  const db = openProjectDb(folder)
  const now = Date.now()
  const id = randomUUID()
  db.prepare('INSERT INTO project (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)').run(
    id,
    input.name,
    now,
    now,
  )

  // Start every project with one empty chooser node on the canvas, so the user lands on the
  // "Link a ComfyUI Workflow / Generate with Fal" prompt instead of a blank board.
  const frame = createEmptyFrame()
  addFrameItem(frame.id, 80, 80)

  const project: Project = { id, name: input.name, path: folder, createdAt: now, updatedAt: now }
  currentProject = project
  recordRecent({ name: project.name, path: folder })
  return project
}

export function openProject(selected: string): Project {
  // Resolve the real project folder — it may be the picked one, or a level down after
  // an unzip wrapped it (see resolveProjectFolder). A valid project.db is the real
  // signal, so hand-renamed or legacy `.storyline` folders open too.
  const folder = resolveProjectFolder(selected)
  if (!folder) {
    throw new Error('That folder is not an Inline Studio project (no project.db found).')
  }
  openProjectDb(folder)
  // Make sure media subdirs exist even for hand-copied projects.
  for (const sub of SUBDIRS) {
    const p = join(folder, sub)
    if (!existsSync(p)) mkdirSync(p, { recursive: true })
  }
  const project = loadProjectRow(folder)
  currentProject = project
  recordRecent({ name: project.name, path: folder })
  // Catch up media imported before posters/transcodes/waveforms existed (background, best-effort).
  backfillVideoAssets()
  backfillAudioAssets()
  backfillImageAssets()
  return project
}

/**
 * The actual project folder for a user-picked `folder`: the folder itself when it holds
 * a `project.db`, else one level down. After unzipping an exported project, extractors
 * (notably Windows Explorer) wrap the exported `Foo.inlinestudio/` inside another folder
 * named after the zip — so the picked folder often *contains* the real project as a
 * child rather than being it. Prefers a `*.inlinestudio`/`.storyline` child, then any
 * child, that contains a `project.db`. Returns null if none is found.
 */
export function resolveProjectFolder(folder: string): string | null {
  try {
    if (existsSync(join(folder, 'project.db'))) return folder
    const children = readdirSync(folder, { withFileTypes: true }).filter((e) => e.isDirectory())
    const hasDb = (name: string): boolean => existsSync(join(folder, name, 'project.db'))
    const preferred = children.find((e) => isProjectExt(e.name) && hasDb(e.name))
    if (preferred) return join(folder, preferred.name)
    const any = children.find((e) => hasDb(e.name))
    return any ? join(folder, any.name) : null
  } catch {
    return null
  }
}

/**
 * Heuristic for the open dialog: does this dir hold (or wrap) an Inline Studio project?
 * A valid `project.db` is the real signal, so this also accepts legacy `.storyline`
 * folders, folders renamed away from the `.inlinestudio` extension, and unzip-nested ones.
 */
export function isProjectFolder(folder: string): boolean {
  return resolveProjectFolder(folder) !== null
}

export function getCurrentProject(): Project | null {
  return currentProject
}
