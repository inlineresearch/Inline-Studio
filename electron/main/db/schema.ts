/**
 * SQLite schema for a project's `project.db`. The DB is the source of truth for
 * a project — "save" is implicit. Bumping SCHEMA_VERSION + adding a migration is
 * how the schema evolves (see runMigrations).
 */
import type BetterSqlite3 from 'better-sqlite3'

export const SCHEMA_VERSION = 14

const SCHEMA_SQL = `
CREATE TABLE IF NOT EXISTS project (
  id          TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  created_at  INTEGER NOT NULL,
  updated_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sequences (
  id          TEXT PRIMARY KEY,
  project_id  TEXT NOT NULL,
  name        TEXT NOT NULL,
  position    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS frames (
  id                   TEXT PRIMARY KEY,
  sequence_id          TEXT NOT NULL,
  name                 TEXT NOT NULL,
  kind                 TEXT NOT NULL,
  position             INTEGER NOT NULL,
  input_asset_id       TEXT,
  hero_take_id         TEXT,
  provider             TEXT NOT NULL DEFAULT 'comfy',
  model_id             TEXT,
  params               TEXT NOT NULL DEFAULT '{}',
  workflow_template_id TEXT,
  comfy_workflow_name  TEXT,
  comfy_workflow_ready INTEGER NOT NULL DEFAULT 0,
  created_at           INTEGER NOT NULL,
  updated_at           INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS takes (
  id              TEXT PRIMARY KEY,
  frame_id         TEXT NOT NULL,
  file_path       TEXT NOT NULL,
  kind            TEXT NOT NULL,
  params          TEXT NOT NULL,
  comfy_prompt_id TEXT,
  created_at      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS frame_inputs (
  id             TEXT PRIMARY KEY,
  frame_id        TEXT NOT NULL,
  asset_id       TEXT,
  source_frame_id TEXT,
  position       INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS asset_folders (
  id          TEXT PRIMARY KEY,
  project_id  TEXT NOT NULL,
  name        TEXT NOT NULL,
  parent_id   TEXT,
  created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS assets (
  id           TEXT PRIMARY KEY,
  project_id   TEXT NOT NULL,
  folder_id    TEXT,
  name         TEXT NOT NULL,
  file_path    TEXT NOT NULL,
  kind         TEXT NOT NULL,
  thumb_path   TEXT,
  preview_path TEXT,
  created_at   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS moodboard_items (
  id          TEXT PRIMARY KEY,
  project_id  TEXT NOT NULL,
  type        TEXT NOT NULL DEFAULT 'asset',
  asset_id    TEXT,
  frame_id     TEXT,
  parent_id   TEXT,
  data        TEXT,
  x           REAL NOT NULL,
  y           REAL NOT NULL,
  width       REAL NOT NULL,
  height      REAL NOT NULL,
  rotation    REAL NOT NULL DEFAULT 0,
  z_index     INTEGER NOT NULL DEFAULT 0,
  created_at  INTEGER NOT NULL DEFAULT 0,
  updated_at  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS moodboard_connectors (
  id           TEXT PRIMARY KEY,
  project_id   TEXT NOT NULL,
  from_item_id TEXT NOT NULL,
  to_item_id   TEXT NOT NULL,
  label        TEXT,
  data         TEXT,
  created_at   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS workflow_templates (
  id          TEXT PRIMARY KEY,
  project_id  TEXT NOT NULL,
  name        TEXT NOT NULL,
  graph       TEXT NOT NULL,
  params      TEXT NOT NULL
);

-- In-flight fal generations, so a run survives an app restart: on reopen we re-poll each
-- request id and finish it (download the take) or resume its progress. Cleared on completion.
CREATE TABLE IF NOT EXISTS pending_generation (
  id           TEXT PRIMARY KEY,
  frame_id     TEXT NOT NULL,
  model_id     TEXT NOT NULL,
  endpoint     TEXT NOT NULL,
  request_id   TEXT NOT NULL,
  status_url   TEXT NOT NULL,
  response_url TEXT NOT NULL,
  params       TEXT NOT NULL,
  created_at   INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_frames_sequence ON frames(sequence_id);
CREATE INDEX IF NOT EXISTS idx_takes_frame ON takes(frame_id);
CREATE INDEX IF NOT EXISTS idx_frame_inputs_frame ON frame_inputs(frame_id);
CREATE INDEX IF NOT EXISTS idx_assets_project ON assets(project_id);
CREATE INDEX IF NOT EXISTS idx_assets_folder ON assets(folder_id);
CREATE INDEX IF NOT EXISTS idx_asset_folders_parent ON asset_folders(parent_id);
CREATE INDEX IF NOT EXISTS idx_moodboard_items_project ON moodboard_items(project_id);
CREATE INDEX IF NOT EXISTS idx_moodboard_connectors_project ON moodboard_connectors(project_id);
`

/** Create tables (idempotent) and stamp the schema version. */
export function applySchema(db: BetterSqlite3.Database): void {
  db.pragma('journal_mode = WAL')
  db.pragma('foreign_keys = ON')
  const fromVersion = db.pragma('user_version', { simple: true }) as number
  // Additive column migrations must run before SCHEMA_SQL, since SCHEMA_SQL builds
  // indexes that reference newly-added columns.
  migrateColumns(db)
  db.exec(SCHEMA_SQL)
  runDataMigrations(db, fromVersion)
  stampVersion(db)
}

/** Data migrations that need the current schema (tables) to already exist. */
function runDataMigrations(db: BetterSqlite3.Database, fromVersion: number): void {
  // v5 → v6: move each frame's single input_asset_id into the new frame_inputs table.
  // Idempotent: skip frames that already have inputs.
  if (fromVersion < 6) {
    db.exec(`
      INSERT INTO frame_inputs (id, frame_id, asset_id, position)
      SELECT lower(hex(randomblob(16))), id, input_asset_id, 0
      FROM frames
      WHERE input_asset_id IS NOT NULL
        AND NOT EXISTS (SELECT 1 FROM frame_inputs si WHERE si.frame_id = frames.id);
    `)
  }
}

/**
 * Additive column migrations for pre-existing projects (no-ops on fresh DBs, where
 * SCHEMA_SQL already builds the current shape). Must run before SCHEMA_SQL so its
 * indexes can reference newly-added columns.
 */
function migrateColumns(db: BetterSqlite3.Database): void {
  // v7 → v8: rename "shot" → "frame" across tables/columns. Must run before SCHEMA_SQL
  // (so its CREATE TABLE IF NOT EXISTS frames doesn't make an empty duplicate) and
  // before the addColumnIfMissing calls below (which now reference the frame_* names).
  migrateRenames(db)

  // v1 → v2: assets.folder_id
  addColumnIfMissing(db, 'assets', 'folder_id', 'TEXT')

  // v2 → v3: moodboard_items gains generic-item columns (the old shape had only
  // asset_id/note/x/y/width/height). moodboard_connectors is created by SCHEMA_SQL.
  addColumnIfMissing(db, 'moodboard_items', 'type', "TEXT NOT NULL DEFAULT 'asset'")
  addColumnIfMissing(db, 'moodboard_items', 'data', 'TEXT')
  addColumnIfMissing(db, 'moodboard_items', 'rotation', 'REAL NOT NULL DEFAULT 0')
  addColumnIfMissing(db, 'moodboard_items', 'z_index', 'INTEGER NOT NULL DEFAULT 0')
  addColumnIfMissing(db, 'moodboard_items', 'created_at', 'INTEGER NOT NULL DEFAULT 0')
  addColumnIfMissing(db, 'moodboard_items', 'updated_at', 'INTEGER NOT NULL DEFAULT 0')

  // v3 → v4: frames gain a source asset reference.
  addColumnIfMissing(db, 'frames', 'input_asset_id', 'TEXT')

  // v4 → v5: frames gain a linked ComfyUI workflow name.
  addColumnIfMissing(db, 'frames', 'comfy_workflow_name', 'TEXT')

  // v5 → v6: frames gain a source asset reference (frame_inputs created by SCHEMA_SQL).
  // (frame_inputs.comfy_workflow handled above; nothing else here.)

  // v6 → v7: moodboard items can be frames/layers/previews on the unified canvas.
  addColumnIfMissing(db, 'moodboard_items', 'frame_id', 'TEXT')
  addColumnIfMissing(db, 'moodboard_items', 'parent_id', 'TEXT')
  // frame_inputs can reference another frame's output (the refine/flow connector).
  addColumnIfMissing(db, 'frame_inputs', 'source_frame_id', 'TEXT')

  // v8 → v9: a flow input has a source_frame_id and NO asset_id, so asset_id must be
  // nullable. Older DBs created the column NOT NULL, which SQLite can't relax via
  // ALTER — rebuild the table when needed. Runs after source_frame_id exists above.
  relaxFrameInputsAssetId(db)

  // v9 → v10: assets gain a Chromium-playable transcode path (for videos in codecs
  // the UI can't decode natively).
  addColumnIfMissing(db, 'assets', 'preview_path', 'TEXT')

  // v10 → v11: frames track whether a real (non-seed) workflow has been captured, so
  // the UI can tell "linked but empty" from "ready to generate".
  addColumnIfMissing(db, 'frames', 'comfy_workflow_ready', 'INTEGER NOT NULL DEFAULT 0')

  // v11 → v12: the director node stores its state on the moodboard item + derives its
  // timeline from connections, so no table change is needed (the old, never-shipped
  // timeline_clips table is simply left unused if a test DB created one).

  // v12 → v13: frames gain a generation provider (comfy|fal) plus, for fal frames, the
  // registry model id + its editable param values. Existing frames default to comfy, so
  // every ComfyUI code path is untouched.
  addColumnIfMissing(db, 'frames', 'provider', "TEXT NOT NULL DEFAULT 'comfy'")
  addColumnIfMissing(db, 'frames', 'model_id', 'TEXT')
  addColumnIfMissing(db, 'frames', 'params', "TEXT NOT NULL DEFAULT '{}'")
}

/** Rebuild frame_inputs to drop a legacy NOT NULL on asset_id. Idempotent. */
function relaxFrameInputsAssetId(db: BetterSqlite3.Database): void {
  if (!tableExists(db, 'frame_inputs')) return
  const cols = db.pragma('table_info(frame_inputs)') as Array<{ name: string; notnull: number }>
  const assetCol = cols.find((c) => c.name === 'asset_id')
  if (!assetCol || assetCol.notnull === 0) return // already nullable (or absent)
  db.transaction(() => {
    db.exec(`
      CREATE TABLE frame_inputs_new (
        id              TEXT PRIMARY KEY,
        frame_id        TEXT NOT NULL,
        asset_id        TEXT,
        source_frame_id TEXT,
        position        INTEGER NOT NULL
      );
      INSERT INTO frame_inputs_new (id, frame_id, asset_id, source_frame_id, position)
        SELECT id, frame_id, asset_id, source_frame_id, position FROM frame_inputs;
      DROP TABLE frame_inputs;
      ALTER TABLE frame_inputs_new RENAME TO frame_inputs;
    `)
  })()
}

function tableExists(db: BetterSqlite3.Database, table: string): boolean {
  return !!db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name=?").get(table)
}

function renameColumnIfExists(
  db: BetterSqlite3.Database,
  table: string,
  oldCol: string,
  newCol: string,
): void {
  if (!tableExists(db, table)) return
  const cols = db.pragma(`table_info(${table})`) as Array<{ name: string }>
  const hasOld = cols.some((c) => c.name === oldCol)
  const hasNew = cols.some((c) => c.name === newCol)
  if (hasOld && !hasNew) db.exec(`ALTER TABLE ${table} RENAME COLUMN ${oldCol} TO ${newCol}`)
}

/**
 * v7 → v8: the "shot" domain was renamed to "frame". Rename the existing tables,
 * their shot_id columns, and the moodboard 'shot' item-type value in place so
 * existing projects keep their data. All guarded, so it's a no-op on fresh DBs and
 * idempotent on already-migrated ones.
 */
function migrateRenames(db: BetterSqlite3.Database): void {
  if (tableExists(db, 'shots') && !tableExists(db, 'frames')) {
    db.exec('ALTER TABLE shots RENAME TO frames')
  }
  if (tableExists(db, 'shot_inputs') && !tableExists(db, 'frame_inputs')) {
    db.exec('ALTER TABLE shot_inputs RENAME TO frame_inputs')
  }
  renameColumnIfExists(db, 'frame_inputs', 'shot_id', 'frame_id')
  renameColumnIfExists(db, 'frame_inputs', 'source_shot_id', 'source_frame_id')
  renameColumnIfExists(db, 'takes', 'shot_id', 'frame_id')
  renameColumnIfExists(db, 'moodboard_items', 'shot_id', 'frame_id')
  renameColumnIfExists(db, 'timeline_clips', 'shot_id', 'frame_id')
  if (tableExists(db, 'moodboard_items')) {
    db.exec("UPDATE moodboard_items SET type='frame' WHERE type='shot'")
  }
  // Drop the now-misnamed indexes; SCHEMA_SQL recreates them with frame_* names.
  db.exec(
    'DROP INDEX IF EXISTS idx_shots_sequence;' +
      'DROP INDEX IF EXISTS idx_takes_shot;' +
      'DROP INDEX IF EXISTS idx_shot_inputs_shot;',
  )
}

function addColumnIfMissing(
  db: BetterSqlite3.Database,
  table: string,
  column: string,
  definition: string,
): void {
  const exists = db
    .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name=?")
    .get(table)
  if (!exists) return
  const cols = db.pragma(`table_info(${table})`) as Array<{ name: string }>
  if (!cols.some((c) => c.name === column)) {
    db.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${definition}`)
  }
}

function stampVersion(db: BetterSqlite3.Database): void {
  const current = db.pragma('user_version', { simple: true }) as number
  if (current < SCHEMA_VERSION) {
    db.pragma(`user_version = ${SCHEMA_VERSION}`)
  }
}
