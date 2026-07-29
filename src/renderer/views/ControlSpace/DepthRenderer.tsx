/**
 * Renders the Control Space characters to a grayscale depth map (near = bright), the control image a
 * depth ControlNet expects, and hands back a PNG blob. Fires when `nonce` changes and `enabled` (the
 * output kind is "depth"). The scene itself is built by `depthScene.ts`; this owns only the
 * render-target round trip, which is why it is the browser-gated half.
 */
import { useEffect } from 'react'
import { useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { buildDepthScene } from './depthScene'
import { type Vec3 } from './skeleton'

export function DepthRenderer({
  characters,
  nonce,
  enabled,
  aspect,
  onRendered,
}: {
  characters: Vec3[][]
  nonce: number
  enabled: boolean
  aspect: number
  onRendered: (blob: Blob) => void
}): null {
  const gl = useThree((s) => s.gl)
  const camera = useThree((s) => s.camera)

  useEffect(() => {
    if (nonce === 0 || !enabled) return
    // Render at the OUTPUT aspect (not the wide viewport) so depth isn't stretched at gen time.
    const w = aspect >= 1 ? 768 : Math.max(1, Math.round(768 * aspect))
    const h = aspect >= 1 ? Math.max(1, Math.round(768 / aspect)) : 768
    const renderCam = (camera as THREE.PerspectiveCamera).clone()
    renderCam.aspect = aspect
    renderCam.updateProjectionMatrix()

    if (!characters.length) return
    const { scene, dispose } = buildDepthScene(characters, renderCam)

    const target = new THREE.WebGLRenderTarget(w, h)
    const prevTarget = gl.getRenderTarget()
    const prevClear = gl.getClearColor(new THREE.Color()).clone()
    const prevAlpha = gl.getClearAlpha()
    gl.setRenderTarget(target)
    gl.setClearColor(0x000000, 1) // background = far = black
    gl.clear()
    gl.render(scene, renderCam)
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
    dispose()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nonce])
  return null
}
