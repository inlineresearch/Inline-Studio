/**
 * Diagnostic: reproduce exactly how the editor projects 3D joints to the 2D OpenPose control map,
 * so we can see (headlessly, no GPU) whether "Turn around" changes the map and whether characters
 * land inside the rendered frame (an off-frame pose -> near-black map -> random generation).
 */
import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import { KEYPOINTS, STANDING, rotatePoseY, translatePose, type Vec3 } from './skeleton'

/** The editor's default view: camera at (0,1.2,3.2), fov 45, looking at (0,0.9,0). */
function makeCamera(aspect: number): THREE.PerspectiveCamera {
  const cam = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000)
  cam.position.set(0, 1.2, 3.2)
  cam.lookAt(0, 0.9, 0)
  cam.updateMatrixWorld(true)
  cam.updateProjectionMatrix()
  return cam
}

function outputSize(aspect: number): { w: number; h: number } {
  return aspect >= 1
    ? { w: 768, h: Math.round(768 / aspect) }
    : { w: Math.round(768 * aspect), h: 768 }
}

interface P2 {
  x: number
  y: number
  inFrame: boolean
}

function project(
  joints: readonly Vec3[],
  cam: THREE.PerspectiveCamera,
  w: number,
  h: number,
): P2[] {
  const v = new THREE.Vector3()
  return joints.map(([x, y, z]) => {
    v.set(x, y, z).project(cam)
    const px = (v.x * 0.5 + 0.5) * w
    const py = (1 - (v.y * 0.5 + 0.5)) * h
    return { x: px, y: py, inFrame: v.z < 1 && px >= 0 && px <= w && py >= 0 && py <= h }
  })
}

function meanShift(a: P2[], b: P2[]): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += Math.hypot(a[i].x - b[i].x, a[i].y - b[i].y)
  return s / a.length
}

const framed = (p: P2[]): string => `${p.filter((q) => q.inFrame).length}/${p.length} in-frame`

describe('Control Space projection diagnostics', () => {
  it('reports whether Turn around changes the projected map (symmetric vs asymmetric)', () => {
    const aspect = 1
    const cam = makeCamera(aspect)
    const { w, h } = outputSize(aspect)

    const standing = project(STANDING, cam, w, h)
    const standingTurned = project(rotatePoseY(STANDING, Math.PI), cam, w, h)

    // An asymmetric pose: raise the left wrist/elbow out to the side.
    const armOut = STANDING.map((v) => [...v] as Vec3)
    armOut[6] = [0.5, 1.45, 0] // l-elbow out
    armOut[7] = [0.75, 1.45, 0] // l-wrist out
    const armOutP = project(armOut, cam, w, h)
    const armOutTurned = project(rotatePoseY(armOut, Math.PI), cam, w, h)

    console.log('\n=== TURN AROUND (front camera, 768x768) ===')
    console.log(`standing:            ${framed(standing)}`)
    console.log(
      `standing turned:     mean joint shift = ${meanShift(standing, standingTurned).toFixed(1)} px  (symmetric pose -> ~mirror, small)`,
    )
    console.log(
      `arm-out l-wrist px:  ${armOutP[7].x.toFixed(0)}  -> turned: ${armOutTurned[7].x.toFixed(0)}  (should cross the body)`,
    )
    console.log(
      `arm-out turned:      mean joint shift = ${meanShift(armOutP, armOutTurned).toFixed(1)} px`,
    )

    // Turn around DOES change the map (mirror + limb-color swap = a back-view OpenPose), even for a
    // symmetric pose; an asymmetric pose flips its arms across the body. So "turn around not working"
    // is the model not following front/back from OpenPose - not a geometry/save bug.
    expect(meanShift(standing, standingTurned)).toBeGreaterThan(30)
    expect(meanShift(armOutP, armOutTurned)).toBeGreaterThan(80)
  })

  it('reports whether 1 and 2 characters stay inside the frame at each aspect', () => {
    for (const aspect of [1, 3 / 4, 16 / 9]) {
      const cam = makeCamera(aspect)
      const { w, h } = outputSize(aspect)
      const one = project(STANDING, cam, w, h)
      // The editor auto-spaces a 2nd character at +0.8 in x.
      const twoA = project(STANDING, cam, w, h)
      const twoB = project(translatePose(STANDING, 0.8, 0, 0), cam, w, h)
      const label = aspect === 1 ? '1:1' : aspect < 1 ? '3:4' : '16:9'
      console.log(
        `\naspect ${label} (${w}x${h}): 1 char ${framed(one)} | 2 chars A ${framed(twoA)} B ${framed(twoB)}`,
      )
    }
    expect(KEYPOINTS.length).toBe(18)
  })

  it('shows where the 2nd character lands vs the frame edge (x in pixels)', () => {
    const cam = makeCamera(1)
    const { w } = outputSize(1)
    const secondChar = project(translatePose(STANDING, 0.8, 0, 0), cam, w, w)
    const xs = secondChar.map((p) => p.x)
    console.log(
      `\n2nd char x-range at 1:1 768-wide: [${Math.min(...xs).toFixed(0)}, ${Math.max(...xs).toFixed(0)}]  (frame is 0..768)`,
    )
    expect(true).toBe(true)
  })
})

describe('full-body framing at portrait aspect (832x1216)', () => {
  it('reports where head and feet land in the map', () => {
    const aspect = 832 / 1216 // ~0.684 - the user's gen size
    const cam = makeCamera(aspect)
    const { w, h } = outputSize(aspect)
    const p = project(STANDING, cam, w, h)
    const head = p[0] // nose
    const feet = [p[10], p[13]] // ankles
    const inFrame = p.filter((q) => q.inFrame).length
    console.log(`\naspect ${aspect.toFixed(3)} map ${w}x${h}:`)
    console.log(`  head y = ${head.y.toFixed(0)} (top margin ${((head.y / h) * 100).toFixed(0)}%)`)
    console.log(`  feet y = ${Math.max(feet[0].y, feet[1].y).toFixed(0)} (bottom edge at ${h})`)
    console.log(
      `  figure spans ${(((Math.max(feet[0].y, feet[1].y) - head.y) / h) * 100).toFixed(0)}% of the map height`,
    )
    console.log(`  ${inFrame}/18 joints in-frame`)
    expect(inFrame).toBe(18)
  })
})
