import { describe, expect, it, vi } from 'vitest'
import {
  centroid,
  drawOpenPose,
  KEYPOINT_COUNT,
  KEYPOINTS,
  LIMBS,
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
