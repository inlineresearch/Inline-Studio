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

/** A rest pose: standing, facing +Z, Y up, roughly 1.7 units tall. Indexed by KEYPOINTS order.
 * The face keypoints sit a real head-radius in front of the ears (not a few cm), so the head's
 * forward vector - which drives both facing detection and the depth proxy's nose - is unambiguous. */
export const STANDING: readonly Vec3[] = [
  [0, 1.62, 0.12],
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
  [-0.03, 1.66, 0.1],
  [0.03, 1.66, 0.1],
  [-0.07, 1.64, 0.02],
  [0.07, 1.64, 0.02],
]

/** Sitting: hips + knees bent forward, feet under the knees. */
export const SITTING: readonly Vec3[] = [
  [0, 1.12, 0.12],
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
  [-0.03, 1.16, 0.1],
  [0.03, 1.16, 0.1],
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

/** The front-of-face keypoints (nose, right eye, left eye). An OpenPose ControlNet reads front vs
 * back almost entirely from whether these are present, so they must be culled when the head faces
 * away - otherwise a turned-around character still renders looking at the camera. */
const FACE_FRONT: readonly number[] = [0, 14, 15]

/** Every keypoint that belongs to the head, not the body. */
export const HEAD_POINTS: readonly number[] = [0, 14, 15, 16, 17]

function norm([x, y, z]: Vec3): Vec3 {
  const len = Math.hypot(x, y, z) || 1
  return [x / len, y / len, z / len]
}

/**
 * The head's center and unit forward (gaze) direction: the ear midpoint is the skull center and
 * ear-midpoint -> nose is forward, falling back to the neck when the ears are degenerate. Shared by
 * the facing math and the depth body proxy, which both need to know where the face points.
 */
export function headFrame(joints: readonly Vec3[]): { center: Vec3; forward: Vec3 } | null {
  const nose = joints[0]
  const neck = joints[1]
  const rEar = joints[16]
  const lEar = joints[17]
  if (!nose || !neck) return null
  const center: Vec3 =
    rEar && lEar
      ? [(rEar[0] + lEar[0]) / 2, (rEar[1] + lEar[1]) / 2, (rEar[2] + lEar[2]) / 2]
      : neck
  const fwd: Vec3 = [nose[0] - center[0], nose[1] - center[1], nose[2] - center[2]]
  if (Math.hypot(...fwd) < 1e-6) return null
  return { center, forward: norm(fwd) }
}

/** cos of the angle between the head's forward direction and the direction to the camera: 1 = looking
 * straight at it, 0 = side-on profile, -1 = looking directly away. */
export function facingCos(joints: readonly Vec3[], camPos: Vec3): number | null {
  const head = headFrame(joints)
  const nose = joints[0]
  if (!head || !nose) return null
  const [tx, ty, tz] = norm([camPos[0] - nose[0], camPos[1] - nose[1], camPos[2] - nose[2]])
  return head.forward[0] * tx + head.forward[1] * ty + head.forward[2] * tz
}

/**
 * Which keypoints the head occludes by facing away from the camera - a boolean per KEYPOINTS index
 * (true = drop it). When the face points away from the camera those keypoints are on the hidden
 * side, so we drop the nose + eyes and keep the ears, which is what OpenPose sees from behind.
 * Purely geometric, so it holds for any character orientation or camera angle.
 */
export function faceOcclusion(joints: readonly Vec3[], camPos: Vec3): boolean[] {
  const cull = new Array(KEYPOINT_COUNT).fill(false) as boolean[]
  const cos = facingCos(joints, camPos)
  if (cos !== null && cos < 0) for (const i of FACE_FRONT) cull[i] = true
  return cull
}

export type Facing = 'front' | 'three-quarter' | 'profile' | 'back'

/** How the character is turned relative to the camera. `null` when the head is degenerate. */
export function facingLabel(joints: readonly Vec3[], camPos: Vec3): Facing | null {
  const cos = facingCos(joints, camPos)
  if (cos === null) return null
  if (cos > 0.5) return 'front'
  if (cos > 0.15) return 'three-quarter'
  if (cos > -0.15) return 'profile'
  return 'back'
}

/**
 * The prompt text a facing needs, per character. A face-less OpenPose skeleton is *ambiguous*, not
 * back-facing - real detections lose the face to blur and distance too - so the model falls back on
 * its prior (a visible face) and renders the head turned back over the shoulder. The control image
 * has no channel that can say "away"; the text encoder does, so the facing is stated there.
 * Front/three-quarter need nothing: they are what the prior already assumes.
 */
const FACING_PROMPT: Record<Facing, { positive: string; negative: string }> = {
  front: { positive: '', negative: '' },
  'three-quarter': { positive: '', negative: '' },
  profile: {
    positive: 'seen in profile, side view, head facing sideways',
    negative: 'facing the camera, front view',
  },
  back: {
    positive: 'seen from behind, back view, back of the head, facing away from the camera',
    negative: 'face visible, looking at the camera, head turned back over the shoulder, front view',
  },
}

/**
 * The prompt hint for a whole scene, or null when there is nothing to say. Only emitted when every
 * character shares a facing - a mixed scene has no single true statement, and a wrong global hint is
 * worse than none.
 */
export function facingPromptHint(
  labels: readonly (Facing | null)[],
): { positive: string; negative: string } | null {
  const known = labels.filter((l): l is Facing => l !== null)
  if (!known.length || known.some((l) => l !== known[0])) return null
  const hint = FACING_PROMPT[known[0]]
  return hint.positive ? hint : null
}

/**
 * Distance to the nearest ground-plane point the camera can see - the bottom edge of the frame,
 * which is the closest surface in the map once a floor is rendered. Null when the camera looks at or
 * above the horizon and never sees the ground. The bottom-centre ray is the nearest hit: a corner
 * ray points less steeply downward once normalized, so it lands farther away.
 */
export function nearestGroundHit(
  camPos: Vec3,
  forward: Vec3,
  up: Vec3,
  fovDeg: number,
  groundY = 0,
): number | null {
  const t = Math.tan((fovDeg * Math.PI) / 360)
  const dir = norm([forward[0] - up[0] * t, forward[1] - up[1] * t, forward[2] - up[2] * t])
  if (dir[1] >= -1e-6) return null
  const distance = (groundY - camPos[1]) / dir[1]
  return distance > 0 ? distance : null
}

/**
 * The camera-distance range a depth map should span, plus where to put the backdrop.
 *
 * A depth ControlNet is trained on monocular depth estimates of real photos: dense, full-frame, and
 * using the whole 0-255 range. So `near` is the nearest surface actually in frame (the floor at the
 * bottom edge, else the closest body), and `far` sits a little past the backdrop so the background
 * lands on a dark grey rather than the pure black that reads as "no depth here".
 */
export function depthRange(
  characters: readonly (readonly Vec3[])[],
  camPos: Vec3,
  groundNear: number | null = null,
): { near: number; far: number; backdrop: number } {
  let dmin = Infinity
  let dmax = -Infinity
  for (const joints of characters) {
    for (const [x, y, z] of joints) {
      const d = Math.hypot(x - camPos[0], y - camPos[1], z - camPos[2])
      if (d < dmin) dmin = d
      if (d > dmax) dmax = d
    }
  }
  if (!isFinite(dmin)) return { near: 0.05, far: 1, backdrop: 0.8 }
  const skin = 0.15 // a torso capsule radius: the body surface sits this far in front of its joints
  const backdrop = dmax + skin + 1.2
  const near = Math.max(0.05, Math.min(dmin - skin, groundNear ?? Infinity))
  return { near, far: backdrop + 0.15 * (backdrop - near), backdrop }
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
