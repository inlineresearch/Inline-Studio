/**
 * Durable record of in-flight fal generations. A fal request keeps running server-side even if
 * the app is closed, so we persist just enough (the request id + its status/response URLs + the
 * params needed to save the take) to re-poll and finish it when the project is reopened.
 *
 * One row per frame — a fresh run for a frame replaces its previous pending row.
 */
import { randomUUID } from 'node:crypto'
import { getDb } from '../db'

export interface PendingRun {
  id: string
  frameId: string
  modelId: string
  endpoint: string
  requestId: string
  statusUrl: string
  responseUrl: string
  params: Record<string, unknown>
  createdAt: number
}

interface PendingRow {
  id: string
  frame_id: string
  model_id: string
  endpoint: string
  request_id: string
  status_url: string
  response_url: string
  params: string
  created_at: number
}

function rowToRun(r: PendingRow): PendingRun {
  let params: Record<string, unknown> = {}
  try {
    params = JSON.parse(r.params) as Record<string, unknown>
  } catch {
    params = {}
  }
  return {
    id: r.id,
    frameId: r.frame_id,
    modelId: r.model_id,
    endpoint: r.endpoint,
    requestId: r.request_id,
    statusUrl: r.status_url,
    responseUrl: r.response_url,
    params,
    createdAt: r.created_at,
  }
}

/** Persist an in-flight run for `frameId`, replacing any previous pending row for that frame. */
export function recordPending(run: Omit<PendingRun, 'id' | 'createdAt'>): void {
  const db = getDb()
  db.prepare('DELETE FROM pending_generation WHERE frame_id = ?').run(run.frameId)
  db.prepare(
    `INSERT INTO pending_generation
       (id, frame_id, model_id, endpoint, request_id, status_url, response_url, params, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run(
    randomUUID(),
    run.frameId,
    run.modelId,
    run.endpoint,
    run.requestId,
    run.statusUrl,
    run.responseUrl,
    JSON.stringify(run.params ?? {}),
    Date.now(),
  )
}

/** Forget the pending run for a frame (call on completion, error, or cancel). */
export function deletePendingByFrame(frameId: string): void {
  getDb().prepare('DELETE FROM pending_generation WHERE frame_id = ?').run(frameId)
}

/** Every persisted in-flight run, oldest first. */
export function listPending(): PendingRun[] {
  const rows = getDb()
    .prepare('SELECT * FROM pending_generation ORDER BY created_at')
    .all() as PendingRow[]
  return rows.map(rowToRun)
}
