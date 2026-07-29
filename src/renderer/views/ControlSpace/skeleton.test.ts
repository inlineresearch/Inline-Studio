import { describe, expect, it, vi } from 'vitest'
import {
  centroid,
  depthRange,
  drawOpenPose,
  faceOcclusion,
  facingLabel,
  facingPromptHint,
  headFrame,
  KEYPOINT_COUNT,
  KEYPOINTS,
  LIMBS,
  nearestGroundHit,
  POSE_COLORS,
  PRESETS,
  rotatePoseY,
  STANDING,
  translatePose,
  type Point2D,
  type Vec3,
} from './skeleton'

describe('OpenPose skeleton model', () => {
  it('has 18 keypoints and one color per keypoint', () => {
    expect(KEYPOINT_COUNT).toBe(18)
    expect(KEYPOINTS).toHaveLength(18)
    expect(POSE_COLORS).toHaveLength(18)
  })

  it('every limb connects valid keypoint indices', () => {
    for (const [a, b] of LIMBS) {
      expect(a).toBeGreaterThanOrEqual(0)
      expect(b).toBeGreaterThanOrEqual(0)
      expect(a).toBeLessThan(KEYPOINT_COUNT)
      expect(b).toBeLessThan(KEYPOINT_COUNT)
    }
  })

  it('every preset defines all 18 joints', () => {
    for (const pose of Object.values(PRESETS)) expect(pose).toHaveLength(18)
  })
})

describe('pose transforms', () => {
  const approx = (a: readonly Vec3[], b: readonly Vec3[]): void => {
    a.forEach(([x, y, z], i) => {
      expect(x).toBeCloseTo(b[i][0], 6)
      expect(y).toBeCloseTo(b[i][1], 6)
      expect(z).toBeCloseTo(b[i][2], 6)
    })
  }

  it('centroid averages the joints', () => {
    const [cx, cy, cz] = centroid([
      [0, 0, 0],
      [2, 4, 6],
    ])
    expect([cx, cy, cz]).toEqual([1, 2, 3])
  })

  it('a full turn (2pi) is a no-op and never changes Y', () => {
    approx(rotatePoseY(STANDING, Math.PI * 2), STANDING)
    rotatePoseY(STANDING, 1.234).forEach(([, y], i) => expect(y).toBeCloseTo(STANDING[i][1], 6))
  })

  it('a half turn mirrors x and z about the centroid (front/back)', () => {
    const [cx, , cz] = centroid(STANDING)
    rotatePoseY(STANDING, Math.PI).forEach(([x, , z], i) => {
      expect(x).toBeCloseTo(2 * cx - STANDING[i][0], 6)
      expect(z).toBeCloseTo(2 * cz - STANDING[i][2], 6)
    })
  })

  it('translatePose shifts every joint by the offset', () => {
    translatePose(STANDING, 1, -2, 3).forEach(([x, y, z], i) => {
      expect(x).toBeCloseTo(STANDING[i][0] + 1, 6)
      expect(y).toBeCloseTo(STANDING[i][1] - 2, 6)
      expect(z).toBeCloseTo(STANDING[i][2] + 3, 6)
    })
  })
})

describe('faceOcclusion (front/back facing)', () => {
  // STANDING faces +Z (nose is forward of the neck/ears in +Z).
  const NOSE = 0
  const R_EYE = 14
  const L_EYE = 15
  const R_EAR = 16
  const L_EAR = 17

  it('keeps the face when the camera is in front of it', () => {
    const cull = faceOcclusion(STANDING, [0, 1.5, 5]) // camera at +Z, facing the character's front
    expect(cull[NOSE]).toBe(false)
    expect(cull[R_EYE]).toBe(false)
    expect(cull[L_EYE]).toBe(false)
  })

  it('culls the nose and eyes but keeps the ears when the head faces away', () => {
    const cull = faceOcclusion(STANDING, [0, 1.5, -5]) // camera behind the character
    expect(cull[NOSE]).toBe(true)
    expect(cull[R_EYE]).toBe(true)
    expect(cull[L_EYE]).toBe(true)
    expect(cull[R_EAR]).toBe(false) // ears stay - what OpenPose sees from behind
    expect(cull[L_EAR]).toBe(false)
  })

  it('turning the character around flips which side the camera sees as the face', () => {
    const turned = rotatePoseY(STANDING, Math.PI) // now faces -Z
    const front = faceOcclusion(turned, [0, 1.5, 5]) // camera at +Z now sees the back of the head
    expect(front[NOSE]).toBe(true)
    const behind = faceOcclusion(turned, [0, 1.5, -5]) // camera at -Z now sees the face
    expect(behind[NOSE]).toBe(false)
  })
})

describe('facing', () => {
  const FRONT: Vec3 = [0, 1.5, 5]
  const BEHIND: Vec3 = [0, 1.5, -5]
  const SIDE: Vec3 = [5, 1.5, 0.12] // level with the nose, so the camera is exactly side-on

  it('the head frame points from the skull center toward the nose', () => {
    const head = headFrame(STANDING)
    expect(head).not.toBeNull()
    expect(head!.forward[2]).toBeGreaterThan(0.8) // STANDING faces +Z
    expect(Math.hypot(...head!.forward)).toBeCloseTo(1, 6)
  })

  it('labels front, back and profile from the camera position', () => {
    expect(facingLabel(STANDING, FRONT)).toBe('front')
    expect(facingLabel(STANDING, BEHIND)).toBe('back')
    expect(facingLabel(STANDING, SIDE)).toBe('profile')
  })

  it('follows the character when it turns around', () => {
    const turned = rotatePoseY(STANDING, Math.PI)
    expect(facingLabel(turned, FRONT)).toBe('back')
    expect(facingLabel(turned, BEHIND)).toBe('front')
  })

  it('a back-facing scene gets a prompt hint; a front-facing one needs none', () => {
    expect(facingPromptHint(['front'])).toBeNull()
    expect(facingPromptHint(['three-quarter'])).toBeNull()
    const hint = facingPromptHint(['back', 'back'])
    expect(hint?.positive).toContain('from behind')
    expect(hint?.negative).toContain('looking at the camera')
  })

  it('says nothing when characters face different ways or the head is degenerate', () => {
    expect(facingPromptHint(['back', 'front'])).toBeNull()
    expect(facingPromptHint([])).toBeNull()
    expect(facingPromptHint([null])).toBeNull()
  })

  it('agrees with faceOcclusion: a back label means the face keypoints are culled', () => {
    expect(facingLabel(STANDING, BEHIND)).toBe('back')
    expect(faceOcclusion(STANDING, BEHIND)[0]).toBe(true)
    expect(faceOcclusion(STANDING, FRONT)[0]).toBe(false)
  })
})

describe('depth map range', () => {
  const CAM: Vec3 = [0, 1.2, 3.2]
  const FWD: Vec3 = [0, -0.09, -0.99] // the editor's default framing, looking slightly down
  const UP: Vec3 = [0, 1, 0]

  it('finds the ground at the bottom of the frame, and reports none when looking up', () => {
    const hit = nearestGroundHit(CAM, FWD, UP, 45)
    expect(hit).not.toBeNull()
    expect(hit!).toBeGreaterThan(0)
    expect(hit!).toBeLessThan(3.2) // nearer than the character, so it owns the near plane
    expect(nearestGroundHit(CAM, [0, 0.5, -0.87], UP, 45)).toBeNull()
  })

  it('spans nearest visible surface to backdrop, and never lets the backdrop hit pure black', () => {
    const ground = nearestGroundHit(CAM, FWD, UP, 45)
    const { near, far, backdrop } = depthRange([STANDING], CAM, ground)
    expect(near).toBeCloseTo(ground!, 6) // the floor is nearer than the body
    expect(backdrop).toBeGreaterThan(near)
    expect(far).toBeGreaterThan(backdrop) // headroom, so the background is dark grey not 0
    // The shader maps distance -> 1-t; the backdrop must land clear of black.
    const backdropGrey = 1 - (backdrop - near) / (far - near)
    expect(backdropGrey).toBeGreaterThan(0.05)
  })

  it('falls back to the body when there is no ground in frame', () => {
    const { near } = depthRange([STANDING], CAM, null)
    const nearest = Math.min(
      ...STANDING.map(([x, y, z]) => Math.hypot(x - CAM[0], y - CAM[1], z - CAM[2])),
    )
    expect(near).toBeLessThan(nearest) // padded by a body radius so the surface isn't clipped
    expect(near).toBeGreaterThan(nearest - 0.3)
  })

  it('degenerate input does not produce a NaN range', () => {
    const { near, far } = depthRange([], CAM)
    expect(Number.isFinite(near)).toBe(true)
    expect(far).toBeGreaterThan(near)
  })
})

describe('drawOpenPose', () => {
  const fakeCtx = () =>
    ({
      fillRect: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      fillStyle: '',
      strokeStyle: '',
      lineWidth: 0,
      lineCap: '',
    }) as unknown as CanvasRenderingContext2D

  const allVisible = (): Point2D[] =>
    Array.from({ length: 18 }, (_, i) => ({ x: i * 10, y: i * 5, visible: true }))

  it('clears the background and draws a dot per visible joint and a stroke per drawable limb', () => {
    const ctx = fakeCtx()
    drawOpenPose(ctx, allVisible(), 512, 512)
    expect(ctx.fillRect).toHaveBeenCalledTimes(1) // black background
    expect((ctx.arc as ReturnType<typeof vi.fn>).mock.calls.length).toBe(18) // one dot per joint
    expect((ctx.stroke as ReturnType<typeof vi.fn>).mock.calls.length).toBe(LIMBS.length)
  })

  it('skips limbs and dots for hidden joints', () => {
    const ctx = fakeCtx()
    const pts = allVisible()
    pts[4].visible = false // r-wrist hidden -> its dot and the elbow-wrist limb are skipped
    drawOpenPose(ctx, pts, 512, 512)
    expect((ctx.arc as ReturnType<typeof vi.fn>).mock.calls.length).toBe(17)
    const wristLimbs = LIMBS.filter(([a, b]) => a === 4 || b === 4).length
    expect((ctx.stroke as ReturnType<typeof vi.fn>).mock.calls.length).toBe(
      LIMBS.length - wristLimbs,
    )
  })

  it('can skip the clear to overlay multiple characters', () => {
    const ctx = fakeCtx()
    drawOpenPose(ctx, allVisible(), 512, 512, false)
    expect(ctx.fillRect).not.toHaveBeenCalled()
  })
})
