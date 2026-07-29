/**
 * The depth body proxy: the three.js scene a Control Space depth map is rendered from. React-free so
 * it can be rendered and measured outside the app.
 *
 * Two things make this map usable as a ControlNet input. The bodies have real volume - limb capsules
 * with anatomical thickness, a skull, and a face that protrudes along the head's forward axis, so
 * facing is legible in the pixels (the face self-occludes when the character turns away). And the
 * frame is dense: a ground plane and backdrop mean every pixel carries a depth value, because these
 * models are trained on monocular depth estimates of real photos, never on a silhouette in a void.
 *
 * The depth convention is owned here (not three's MeshDepthMaterial packing), so it is unambiguous.
 */
import * as THREE from 'three'
import { depthRange, headFrame, HEAD_POINTS, LIMBS, nearestGroundHit, type Vec3 } from './skeleton'

const DEPTH_SHADER = {
  vertex: `
    varying float vDist;
    void main() {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      vDist = -mv.z; // positive distance in front of the camera
      gl_Position = projectionMatrix * mv;
    }
  `,
  fragment: `
    uniform float uNear;
    uniform float uFar;
    varying float vDist;
    void main() {
      float t = clamp((vDist - uNear) / max(uFar - uNear, 0.0001), 0.0, 1.0);
      float g = 1.0 - t; // near = bright
      gl_FragColor = vec4(vec3(g), 1.0);
    }
  `,
}

/** Limbs that make up the body; head keypoints are replaced by the skull + face volume below. */
const BODY_LIMBS = LIMBS.filter(([a, b]) => !HEAD_POINTS.includes(a) && !HEAD_POINTS.includes(b))

/** OpenPose has no shoulder-shoulder or hip-hip bone, but a torso is a box, not a V. */
const TORSO_FILL: readonly [number, number][] = [
  [2, 5],
  [8, 11],
]

/** Capsule radius per bone, keyed `a-b` - a torso is not as thin as a forearm. */
const LIMB_RADIUS: Record<string, number> = {
  '1-2': 0.09,
  '1-5': 0.09,
  '2-3': 0.055,
  '3-4': 0.045,
  '5-6': 0.055,
  '6-7': 0.045,
  '1-8': 0.13,
  '1-11': 0.13,
  '8-9': 0.085,
  '9-10': 0.06,
  '11-12': 0.085,
  '12-13': 0.06,
  '2-5': 0.1,
  '8-11': 0.11,
}

/** Sphere radius per joint, so bones meet in a rounded mass instead of a visible seam. */
const JOINT_RADIUS: Record<number, number> = {
  1: 0.085,
  2: 0.085,
  5: 0.085,
  3: 0.05,
  6: 0.05,
  4: 0.04,
  7: 0.04,
  8: 0.095,
  11: 0.095,
  9: 0.06,
  12: 0.06,
  10: 0.045,
  13: 0.045,
}

const SKULL_RADIUS = 0.1

/**
 * Build one character's depth proxy into `scene`. Geometries are collected so the caller can dispose
 * them after the readback.
 */
function buildBody(
  joints: Vec3[],
  mat: THREE.Material,
  scene: THREE.Scene,
  geometries: THREE.BufferGeometry[],
): void {
  const up = new THREE.Vector3(0, 1, 0)
  const a = new THREE.Vector3()
  const b = new THREE.Vector3()
  const dir = new THREE.Vector3()

  const add = (geo: THREE.BufferGeometry): THREE.Mesh => {
    geometries.push(geo)
    const mesh = new THREE.Mesh(geo, mat)
    scene.add(mesh)
    return mesh
  }
  const capsule = (from: THREE.Vector3, to: THREE.Vector3, radius: number): void => {
    dir.subVectors(to, from)
    const len = dir.length()
    if (len < 1e-4) return
    const mesh = add(new THREE.CapsuleGeometry(radius, len, 4, 10))
    mesh.position.copy(from).add(to).multiplyScalar(0.5)
    mesh.quaternion.setFromUnitVectors(up, dir.clone().normalize())
  }

  for (const [i, j] of [...BODY_LIMBS, ...TORSO_FILL]) {
    a.set(...joints[i])
    b.set(...joints[j])
    capsule(a, b, LIMB_RADIUS[`${i}-${j}`] ?? 0.05)
  }
  for (const [index, radius] of Object.entries(JOINT_RADIUS)) {
    add(new THREE.SphereGeometry(radius, 12, 10)).position.set(...joints[Number(index)])
  }

  const head = headFrame(joints)
  if (!head) return
  const center = new THREE.Vector3(...head.center)
  const forward = new THREE.Vector3(...head.forward)

  const neck = new THREE.Vector3(...joints[1])
  capsule(neck, center, 0.05)
  // The head's own up (neck -> skull), so a lying or tilted character's skull is still egg-shaped
  // along the head axis rather than along world Y.
  const headUp = center.clone().sub(neck)
  if (headUp.lengthSq() < 1e-8) headUp.copy(up)
  headUp.normalize()

  const skull = add(new THREE.SphereGeometry(SKULL_RADIUS, 16, 14))
  skull.position.copy(center)
  skull.scale.set(1, 1.15, 1.08)
  skull.quaternion.setFromUnitVectors(up, headUp)

  // The face: a mass that breaks the skull's silhouette forward, plus a nose ridge. Both vanish
  // behind the skull when the head turns away - the whole point of rendering depth from 3D.
  add(new THREE.SphereGeometry(0.055, 12, 10))
    .position.copy(center)
    .addScaledVector(forward, SKULL_RADIUS * 0.85)
    .addScaledVector(headUp, -0.015)
  const nose = add(new THREE.ConeGeometry(0.03, 0.07, 10))
  nose.position.copy(center).addScaledVector(forward, SKULL_RADIUS * 1.2)
  nose.quaternion.setFromUnitVectors(up, forward)

  for (const ear of [16, 17]) {
    if (joints[ear]) add(new THREE.SphereGeometry(0.032, 10, 8)).position.set(...joints[ear])
  }
}

/**
 * A ground plane and a backdrop, so every pixel of the map carries a depth value. Without them the
 * figure floats in a void that is ~78% pure black, which reads as "infinitely far" everywhere and
 * the ControlNet ignores it - monocular depth estimates, what these models are trained on, are dense.
 * The backdrop is billboarded to the render camera so it always covers the frame.
 */
function buildStage(
  scene: THREE.Scene,
  geometries: THREE.BufferGeometry[],
  mat: THREE.Material,
  camera: THREE.PerspectiveCamera,
  backdrop: number,
): void {
  const ground = new THREE.PlaneGeometry(80, 80)
  geometries.push(ground)
  const floor = new THREE.Mesh(ground, mat)
  floor.rotation.x = -Math.PI / 2
  scene.add(floor)

  camera.updateMatrixWorld()
  const wall = new THREE.PlaneGeometry(80, 80)
  geometries.push(wall)
  const back = new THREE.Mesh(wall, mat)
  const forward = camera.getWorldDirection(new THREE.Vector3())
  back.position
    .copy(camera.getWorldPosition(new THREE.Vector3()))
    .addScaledVector(forward, backdrop)
  back.quaternion.copy(camera.getWorldQuaternion(new THREE.Quaternion()))
  scene.add(back)
}

/** The scene a depth map is rendered from: volumetric bodies on a ground plane against a backdrop. */
export function buildDepthScene(
  characters: readonly Vec3[][],
  camera: THREE.PerspectiveCamera,
): { scene: THREE.Scene; dispose: () => void } {
  camera.updateMatrixWorld()
  const cam = camera.getWorldPosition(new THREE.Vector3())
  const fwd = camera.getWorldDirection(new THREE.Vector3())
  const up = camera.up.clone().applyQuaternion(camera.getWorldQuaternion(new THREE.Quaternion()))
  const ground = nearestGroundHit(
    [cam.x, cam.y, cam.z],
    [fwd.x, fwd.y, fwd.z],
    [up.x, up.y, up.z],
    camera.fov,
  )
  const { near, far, backdrop } = depthRange(characters, [cam.x, cam.y, cam.z], ground)
  const mat = new THREE.ShaderMaterial({
    vertexShader: DEPTH_SHADER.vertex,
    fragmentShader: DEPTH_SHADER.fragment,
    uniforms: { uNear: { value: near }, uFar: { value: far } },
  })
  const scene = new THREE.Scene()
  const geometries: THREE.BufferGeometry[] = []
  for (const joints of characters) buildBody(joints, mat, scene, geometries)
  buildStage(scene, geometries, mat, camera, backdrop)
  return {
    scene,
    dispose: () => {
      mat.dispose()
      geometries.forEach((g) => g.dispose())
    },
  }
}
