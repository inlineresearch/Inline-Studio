/**
 * Renders the characters as a grayscale depth map (near = bright), the control image a depth
 * ControlNet expects. The skeleton is thin, so we build a volumetric body offscreen at capture time
 * (a capsule per limb) and render it with a small shader that outputs linear camera-distance,
 * normalized to the scene's own near/far range for good contrast. Fires when `nonce` changes and
 * `enabled` (the output kind is "depth").
 *
 * The depth convention is owned here (not three's MeshDepthMaterial packing), so it is unambiguous;
 * only whether it renders at all is browser-gated, like the rest of the R3F editor.
 */
import { useEffect } from 'react'
import { useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { LIMBS, type Vec3 } from './skeleton'

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

export function DepthRenderer({
  characters,
  nonce,
  enabled,
  onRendered,
}: {
  characters: Vec3[][]
  nonce: number
  enabled: boolean
  onRendered: (blob: Blob) => void
}): null {
  const gl = useThree((s) => s.gl)
  const camera = useThree((s) => s.camera)
  const size = useThree((s) => s.size)

  useEffect(() => {
    if (nonce === 0 || !enabled) return
    const w = 768
    const h = Math.max(1, Math.round((768 * size.height) / size.width))

    // Distance range from the camera to every joint -> adaptive near/far for contrast.
    let dmin = Infinity
    let dmax = -Infinity
    const cam = camera.position
    for (const joints of characters) {
      for (const [x, y, z] of joints) {
        const d = Math.hypot(x - cam.x, y - cam.y, z - cam.z)
        if (d < dmin) dmin = d
        if (d > dmax) dmax = d
      }
    }
    if (!isFinite(dmin)) return
    const mat = new THREE.ShaderMaterial({
      vertexShader: DEPTH_SHADER.vertex,
      fragmentShader: DEPTH_SHADER.fragment,
      uniforms: { uNear: { value: Math.max(0.05, dmin - 0.3) }, uFar: { value: dmax + 0.3 } },
    })

    const scene = new THREE.Scene()
    const up = new THREE.Vector3(0, 1, 0)
    const a = new THREE.Vector3()
    const b = new THREE.Vector3()
    const dir = new THREE.Vector3()
    const geometries: THREE.BufferGeometry[] = []
    for (const joints of characters) {
      for (const [i, j] of LIMBS) {
        a.set(...joints[i])
        b.set(...joints[j])
        dir.subVectors(b, a)
        const len = dir.length()
        if (len < 1e-4) continue
        const geo = new THREE.CapsuleGeometry(0.06, len, 4, 8)
        geometries.push(geo)
        const mesh = new THREE.Mesh(geo, mat)
        mesh.position.copy(a).add(b).multiplyScalar(0.5)
        mesh.quaternion.setFromUnitVectors(up, dir.clone().normalize())
        scene.add(mesh)
      }
    }

    const target = new THREE.WebGLRenderTarget(w, h)
    const prevTarget = gl.getRenderTarget()
    const prevClear = gl.getClearColor(new THREE.Color()).clone()
    const prevAlpha = gl.getClearAlpha()
    gl.setRenderTarget(target)
    gl.setClearColor(0x000000, 1) // background = far = black
    gl.clear()
    gl.render(scene, camera)
    const buf = new Uint8Array(w * h * 4)
    gl.readRenderTargetPixels(target, 0, 0, w, h, buf)
    gl.setRenderTarget(prevTarget)
    gl.setClearColor(prevClear, prevAlpha)

    const canvas = document.createElement('canvas')
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')
    if (ctx) {
      const img = ctx.createImageData(w, h)
      // readRenderTargetPixels is bottom-up; flip vertically into the 2D canvas.
      for (let y = 0; y < h; y++) {
        const srcRow = (h - 1 - y) * w * 4
        const dstRow = y * w * 4
        for (let x = 0; x < w; x++) {
          const s = srcRow + x * 4
          const d = dstRow + x * 4
          img.data[d] = buf[s]
          img.data[d + 1] = buf[s + 1]
          img.data[d + 2] = buf[s + 2]
          img.data[d + 3] = 255
        }
      }
      ctx.putImageData(img, 0, 0)
      canvas.toBlob((blob) => {
        if (blob) onRendered(blob)
      }, 'image/png')
    }

    target.dispose()
    mat.dispose()
    geometries.forEach((g) => g.dispose())
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nonce])
  return null
}
