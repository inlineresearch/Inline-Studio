/**
 * The OpenPose BODY-18 (COCO) skeleton: keypoint order, limb connections, the canonical color
 * palette, a few default 3D poses, and a 2D renderer that draws the skeleton as the control image a
 * pose ControlNet expects. Pure and framework-free so it can be unit tested; the 3D editor projects
 * its rig joints to 2D screen coordinates and hands them to `drawOpenPose`.
 */

/** COCO / OpenPose BODY-18 keypoint order (index === position in every array below). */
export const KEYPOINTS = [
  'nose',
  'neck',
  'r-shoulder',
  'r-elbow',
  'r-wrist',
  'l-shoulder',
  'l-elbow',
  'l-wrist',
  'r-hip',
  'r-knee',
  'r-ankle',
  'l-hip',
  'l-knee',
  'l-ankle',
  'r-eye',
  'l-eye',
  'r-ear',
  'l-ear',
] as const

export const KEYPOINT_COUNT = KEYPOINTS.length

/** The limb connections OpenPose draws, in the canonical order the color palette is keyed to. */
export const LIMBS: readonly [number, number][] = [
  [1, 2],
  [1, 5],
  [2, 3],
  [3, 4],
  [5, 6],
  [6, 7],
  [1, 8],
  [8, 9],
  [9, 10],
  [1, 11],
  [11, 12],
  [12, 13],
  [1, 0],
  [0, 14],
  [14, 16],
  [0, 15],
  [15, 17],
]

/** The canonical OpenPose limb/keypoint color palette (18 hues around the wheel). */
export const POSE_COLORS: readonly string[] = [
  'rgb(255,0,0)',
  'rgb(255,85,0)',
  'rgb(255,170,0)',
  'rgb(255,255,0)',
  'rgb(170,255,0)',
  'rgb(85,255,0)',
  'rgb(0,255,0)',
  'rgb(0,255,85)',
  'rgb(0,255,170)',
  'rgb(0,255,255)',
  'rgb(0,170,255)',
  'rgb(0,85,255)',
  'rgb(0,0,255)',
  'rgb(85,0,255)',
  'rgb(170,0,255)',
  'rgb(255,0,255)',
  'rgb(255,0,170)',
  'rgb(255,0,85)',
]

export type Vec3 = readonly [number, number, number]

/** A rest pose: standing, facing +Z, Y up, roughly 1.7 units tall. Indexed by KEYPOINTS order. */
export const STANDING: readonly Vec3[] = [
  [0, 1.62, 0.05],
  [0, 1.45, 0],
  [-0.18, 1.45, 0],
  [-0.2, 1.15, 0],
  [-0.22, 0.9, 0],
  [0.18, 1.45, 0],
  [0.2, 1.15, 0],
  [0.22, 0.9, 0],
  [-0.1, 1.0, 0],
  [-0.11, 0.55, 0],
  [-0.11, 0.08, 0],
  [0.1, 1.0, 0],
  [0.11, 0.55, 0],
  [0.11, 0.08, 0],
  [-0.03, 1.66, 0.07],
  [0.03, 1.66, 0.07],
  [-0.07, 1.64, 0.02],
  [0.07, 1.64, 0.02],
]

/** Sitting: hips + knees bent forward, feet under the knees. */
export const SITTING: readonly Vec3[] = [
  [0, 1.12, 0.05],
  [0, 0.95, 0],
  [-0.18, 0.95, 0],
  [-0.2, 0.68, 0.05],
  [-0.22, 0.5, 0.2],
  [0.18, 0.95, 0],
  [0.2, 0.68, 0.05],
  [0.22, 0.5, 0.2],
  [-0.1, 0.5, 0],
  [-0.12, 0.45, 0.35],
  [-0.12, 0.08, 0.35],
  [0.1, 0.5, 0],
  [0.12, 0.45, 0.35],
  [0.12, 0.08, 0.35],
  [-0.03, 1.16, 0.07],
  [0.03, 1.16, 0.07],
  [-0.07, 1.14, 0.02],
  [0.07, 1.14, 0.02],
]

/** Lying: the standing pose laid along +X (a 90 degree tilt), head to the right. */
export const LYING: readonly Vec3[] = STANDING.map(([x, y, z]) => [y - 0.85, -x + 0.9, z]) as Vec3[]

export const PRESETS: Record<string, readonly Vec3[]> = {
  standing: STANDING,
  sitting: SITTING,
  lying: LYING,
}

/** Mean position of a pose's joints (its rough center of mass). */
export function centroid(joints: readonly Vec3[]): Vec3 {
  const n = joints.length || 1
  let sx = 0
  let sy = 0
  let sz = 0
  for (const [x, y, z] of joints) {
    sx += x
    sy += y
    sz += z
  }
  return [sx / n, sy / n, sz / n]
}

/** Rotate a pose about its centroid on the vertical (Y) axis. A half turn (Math.PI) faces the
 * character the other way, which is how front/back is posed for the control map. */
export function rotatePoseY(joints: readonly Vec3[], angle: number): Vec3[] {
  const [cx, , cz] = centroid(joints)
  const cos = Math.cos(angle)
  const sin = Math.sin(angle)
  return joints.map(([x, y, z]) => {
    const dx = x - cx
    const dz = z - cz
    return [cx + dx * cos + dz * sin, y, cz - dx * sin + dz * cos] as Vec3
  })
}

/** Shift every joint of a pose by a fixed offset (used to space multiple characters apart). */
export function translatePose(joints: readonly Vec3[], dx: number, dy: number, dz: number): Vec3[] {
  return joints.map(([x, y, z]) => [x + dx, y + dy, z + dz] as Vec3)
}

export interface Point2D {
  /** Pixel coordinates in the control image. */
  x: number
  y: number
  /** False when the joint is behind the camera / culled; its limbs/dot are skipped. */
  visible: boolean
}

/**
 * Draw the skeleton onto a 2D context as an OpenPose control map: black background, colored limb
 * bones, colored keypoint dots. `points` are pixel coordinates (already projected from 3D), one per
 * KEYPOINTS index. Multiple characters are drawn by calling this once per character onto the same
 * context (pass `clear: false` after the first).
 */
export function drawOpenPose(
  ctx: CanvasRenderingContext2D,
  points: Point2D[],
  width: number,
  height: number,
  clear = true,
): void {
  if (clear) {
    ctx.fillStyle = 'black'
    ctx.fillRect(0, 0, width, height)
  }
  const dot = Math.max(3, Math.round(width / 220))
  const bone = Math.max(2, Math.round(width / 260))

  ctx.lineCap = 'round'
  LIMBS.forEach(([a, b], i) => {
    const p = points[a]
    const q = points[b]
    if (!p?.visible || !q?.visible) return
    ctx.strokeStyle = POSE_COLORS[i % POSE_COLORS.length]
    ctx.lineWidth = bone
    ctx.beginPath()
    ctx.moveTo(p.x, p.y)
    ctx.lineTo(q.x, q.y)
    ctx.stroke()
  })
  points.forEach((p, i) => {
    if (!p?.visible) return
    ctx.fillStyle = POSE_COLORS[i % POSE_COLORS.length]
    ctx.beginPath()
    ctx.arc(p.x, p.y, dot, 0, Math.PI * 2)
    ctx.fill()
  })
}
