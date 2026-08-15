/**
 * Control Space: a 3D pose editor in a modal dialog. Pose one or more OpenPose characters, orbit and
 * frame the camera, then render the scene to an OpenPose control image that drives a ControlNet
 * generation. Mounted once at app root (like MediaLightbox); opened per node via `controlSpaceStore`.
 *
 * Depth output and free-fly camera modes build on this same scene (see docs/controlnet-control-space).
 */
import { createPortal } from 'react-dom'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { Grid, OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { useControlSpaceStore } from '../../store/controlSpaceStore'
import { useMoodboardStore } from '../../store/moodboardStore'
import { useAssetStore } from '../../store/assetStore'
import { importFilesToLibrary } from '../../lib/importFiles'
import { CharacterRig } from './CharacterRig'
import { DepthRenderer } from './DepthRenderer'
import {
  centroid,
  drawOpenPose,
  faceOcclusion,
  facingLabel,
  facingPromptHint,
  PRESETS,
  rotatePoseY,
  STANDING,
  translatePose,
  type Facing,
  type Point2D,
  type Vec3,
} from './skeleton'

interface Character {
  id: string
  joints: Vec3[]
}

const rest = (): Vec3[] => STANDING.map((v) => [...v] as Vec3)

/** Read the persisted scene back into characters, tolerating the legacy single-`joints` shape. */
function loadCharacters(scene: {
  joints?: [number, number, number][]
  characters?: { joints: [number, number, number][] }[]
}): Vec3[][] {
  if (scene.characters?.length)
    return scene.characters.map((c) => c.joints.map((v) => [...v] as Vec3))
  if (scene.joints?.length) return [scene.joints.map((v) => [...v] as Vec3)]
  return [rest()]
}

/** Keep the perspective camera's FOV in sync with the slider. */
function CameraRig({ fov }: { fov: number }): null {
  const camera = useThree((s) => s.camera)
  useEffect(() => {
    const cam = camera as THREE.PerspectiveCamera
    if (cam.isPerspectiveCamera) {
      cam.fov = fov
      cam.updateProjectionMatrix()
    }
  }, [camera, fov])
  return null
}

/** Project every character's 3D joints with the live camera and draw them onto one OpenPose control
 * image (a black canvas), then hand back a PNG blob. Fires when `nonce` changes. */
/** Output size for a control map at aspect (w/h), long edge 768. */
function outputSize(aspect: number): { w: number; h: number } {
  return aspect >= 1
    ? { w: 768, h: Math.max(1, Math.round(768 / aspect)) }
    : { w: Math.max(1, Math.round(768 * aspect)), h: 768 }
}

/** A copy of the live camera reprojected at the OUTPUT aspect, so the control map isn't stretched
 * when the generator resizes it to its own (e.g. square) resolution - the pose-distortion fix. */
function outputCamera(camera: THREE.Camera, aspect: number): THREE.PerspectiveCamera {
  const cam = (camera as THREE.PerspectiveCamera).clone()
  cam.aspect = aspect
  cam.updateProjectionMatrix()
  return cam
}

function PoseRenderer({
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
  const camera = useThree((s) => s.camera)

  useEffect(() => {
    if (nonce === 0 || !enabled) return
    const { w, h } = outputSize(aspect)
    const canvas = document.createElement('canvas')
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const cam = outputCamera(camera, aspect)
    const v = new THREE.Vector3()
    const camPos: Vec3 = [cam.position.x, cam.position.y, cam.position.z]
    const project = (joints: Vec3[]): Point2D[] => {
      // Drop the face keypoints the head turns away from the camera, so a back-facing pose reads as
      // back-facing to the ControlNet (not just a mirrored front).
      const cull = faceOcclusion(joints, camPos)
      return joints.map(([x, y, z], idx) => {
        v.set(x, y, z).project(cam)
        return {
          x: (v.x * 0.5 + 0.5) * w,
          y: (1 - (v.y * 0.5 + 0.5)) * h,
          visible: v.z < 1 && !cull[idx],
        }
      })
    }
    // First character clears to black; the rest overlay onto the same map.
    characters.forEach((joints, i) => drawOpenPose(ctx, project(joints), w, h, i === 0))
    canvas.toBlob((blob) => {
      if (blob) onRendered(blob)
    }, 'image/png')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nonce])
  return null
}

const btn =
  'rounded-md border border-border px-2 py-1 text-xs text-zinc-200 hover:bg-panel disabled:opacity-50'

const FACING_TEXT: Record<Facing, string> = {
  front: 'facing the camera',
  'three-quarter': 'three-quarter view',
  profile: 'side profile',
  back: 'facing away (back view)',
}

// Output aspect presets (w/h). Match the pick to the gen node's width/height.
const ASPECTS: { label: string; value: number }[] = [
  { label: '1:1', value: 1 },
  { label: '3:4', value: 3 / 4 },
  { label: '4:3', value: 4 / 3 },
  { label: '16:9', value: 16 / 9 },
]

function RotateIcon({ cw = false }: { cw?: boolean }): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-3.5 w-3.5"
      style={cw ? { transform: 'scaleX(-1)' } : undefined}
    >
      <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
    </svg>
  )
}

// Default export so the mount can `React.lazy` this (three.js stays out of the initial bundle).
export default function ControlSpaceEditor(): React.JSX.Element | null {
  const editingItemId = useControlSpaceStore((s) => s.editingItemId)
  const close = useControlSpaceStore((s) => s.close)
  const [characters, setCharacters] = useState<Character[]>(() => [{ id: 'c0', joints: rest() }])
  const [active, setActive] = useState(0)
  const [selJoint, setSelJoint] = useState<number | null>(null)
  const [fov, setFov] = useState(45)
  const [mapKind, setMapKind] = useState<'pose' | 'depth'>('pose')
  // Output aspect (w/h) of the control map. It must match the gen node's resolution or the generator
  // stretches the map and the pose comes out distorted; default square = Z-Image's 1024x1024 default.
  const [aspect, setAspect] = useState(1)
  const [orbit, setOrbit] = useState(true)
  const [nonce, setNonce] = useState(0)
  const [preview, setPreview] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [facings, setFacings] = useState<(Facing | null)[]>([])
  const [applyHint, setApplyHint] = useState(true)
  const idSeq = useRef(1)
  // 'save' renders then persists the map to the node; 'preview' only refreshes the corner thumbnail.
  const renderMode = useRef<'preview' | 'save'>('preview')

  // The editor remounts each open (the mount gate unmounts it on close), so the saved scene read here
  // seeds the initial camera; live changes are captured back into cameraPose on every orbit change.
  const savedScene = useMemo(
    () => useMoodboardStore.getState().items.find((i) => i.id === editingItemId)?.data.controlScene,
    [editingItemId],
  )
  const initialCamPos = (savedScene?.camera?.position ?? [0, 1.2, 3.2]) as [number, number, number]
  const initialTarget = (savedScene?.camera?.target ?? [0, 0.9, 0]) as [number, number, number]
  const orbitRef = useRef<React.ComponentRef<typeof OrbitControls>>(null)
  const cameraPose = useRef({ position: initialCamPos, target: initialTarget })

  const open = editingItemId !== null
  useEffect(() => {
    if (!open) return
    const scene = savedScene
    idSeq.current = 1
    setCharacters(
      (scene ? loadCharacters(scene) : [rest()]).map((joints) => ({
        id: `c${idSeq.current++}`,
        joints,
      })),
    )
    setFov(scene?.fov ?? 45)
    setMapKind(scene?.output ?? 'pose')
    setAspect(scene?.aspect ?? 1)
    setApplyHint(scene?.applyPromptHint ?? true)
    setActive(0)
    setSelJoint(null)
    setPreview((prev) => {
      if (prev) URL.revokeObjectURL(prev)
      return null
    })
  }, [open, editingItemId, savedScene])

  // Arrow keys nudge the selected joint: Left/Right = X, Up/Down = Y, Shift+Up/Down = Z (depth).
  useEffect(() => {
    if (!open) return
    const step = 0.02
    const onKey = (e: KeyboardEvent): void => {
      if (selJoint === null || e.target instanceof HTMLInputElement) return
      let dx = 0
      let dy = 0
      let dz = 0
      if (e.key === 'ArrowLeft') dx = -step
      else if (e.key === 'ArrowRight') dx = step
      else if (e.key === 'ArrowUp') {
        if (e.shiftKey) dz = -step
        else dy = step
      } else if (e.key === 'ArrowDown') {
        if (e.shiftKey) dz = step
        else dy = -step
      } else return
      e.preventDefault()
      setCharacters((cs) =>
        cs.map((c, i) => {
          if (i !== active) return c
          const joints = c.joints.slice()
          const [x, y, z] = joints[selJoint]
          joints[selJoint] = [x + dx, y + dy, z + dz]
          return { ...c, joints }
        }),
      )
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, active, selJoint])

  // Facing depends on both the pose and the camera, so it is refreshed from each. State only changes
  // when a label actually flips, so orbiting (which fires per frame) doesn't re-render the editor.
  const refreshFacing = useCallback((chars: readonly Vec3[][]): void => {
    const next = chars.map((joints) => facingLabel(joints, cameraPose.current.position))
    setFacings((prev) =>
      prev.length === next.length && prev.every((f, i) => f === next[i]) ? prev : next,
    )
  }, [])
  useEffect(() => {
    refreshFacing(characters.map((c) => c.joints))
  }, [characters, refreshFacing])

  const captureCam = (): void => {
    const c = orbitRef.current
    if (!c) return
    const cam = c.object
    cameraPose.current = {
      position: [cam.position.x, cam.position.y, cam.position.z],
      target: [c.target.x, c.target.y, c.target.z],
    }
    refreshFacing(characters.map((c2) => c2.joints))
  }
  const resetView = (): void => {
    const c = orbitRef.current
    if (!c) return
    c.object.position.set(0, 1.2, 3.2)
    c.target.set(0, 0.9, 0)
    c.update()
  }
  const frameActive = (): void => {
    const c = orbitRef.current
    const char = characters[active] ?? characters[0]
    if (!c || !char) return
    const [cx, cy, cz] = centroid(char.joints)
    c.object.position.set(cx, cy, cz + 3)
    c.target.set(cx, cy, cz)
    c.update()
  }

  const activeChar = characters[active] ?? characters[0]
  const setActiveJoints = (joints: Vec3[]): void =>
    setCharacters((cs) => cs.map((c, i) => (i === active ? { ...c, joints } : c)))

  const addCharacter = (): void => {
    const placed = translatePose(rest(), characters.length * 0.8, 0, 0)
    setCharacters((cs) => [...cs, { id: `c${idSeq.current++}`, joints: placed }])
    setActive(characters.length)
    setSelJoint(null)
  }
  const removeActive = (): void => {
    if (characters.length <= 1) return
    setCharacters((cs) => cs.filter((_, i) => i !== active))
    setActive((a) => Math.max(0, a - 1))
    setSelJoint(null)
  }
  const turn = (angle: number): void => setActiveJoints(rotatePoseY(activeChar.joints, angle))
  const applyPreset = (name: string): void => {
    // Keep the character where it stands: align the preset's centroid to the current one (x/z only).
    const [cx, , cz] = centroid(activeChar.joints)
    const [px, , pz] = centroid(PRESETS[name])
    setActiveJoints(translatePose(PRESETS[name], cx - px, 0, cz - pz))
  }

  const handleRendered = async (blob: Blob): Promise<void> => {
    const url = URL.createObjectURL(blob)
    setPreview((prev) => {
      if (prev) URL.revokeObjectURL(prev)
      return url
    })
    if (renderMode.current !== 'save' || !editingItemId) return
    renderMode.current = 'preview'
    setBusy(true)
    try {
      const file = new File([blob], `control-space-${mapKind}.png`, { type: 'image/png' })
      const [asset] = await importFilesToLibrary([file], null)
      if (!asset) return
      await useAssetStore.getState().load()
      const item = useMoodboardStore.getState().items.find((i) => i.id === editingItemId)
      const scene = {
        characters: characters.map((c) => ({
          joints: c.joints.map((v) => [v[0], v[1], v[2]] as [number, number, number]),
        })),
        fov,
        camera: cameraPose.current,
        output: mapKind,
        aspect,
        facing: facings,
        promptHint: hint,
        applyPromptHint: applyHint,
      }
      await useMoodboardStore.getState().updateItem(editingItemId, {
        data: { ...(item?.data ?? {}), controlAssetId: asset.id, controlScene: scene },
      })
      close()
    } finally {
      setBusy(false)
    }
  }

  const render = (mode: 'preview' | 'save'): void => {
    renderMode.current = mode
    setNonce((n) => n + 1)
  }

  const presetNames = useMemo(() => Object.keys(PRESETS), [])
  const hint = applyHint ? facingPromptHint(facings) : null
  const activeFacing = facings[active] ?? null
  if (!open) return null

  return createPortal(
    <div
      className="fixed inset-0 z-[150] flex items-center justify-center bg-black/70 p-6"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget && !busy) close()
      }}
    >
      <div className="flex h-[min(760px,88vh)] w-[min(1100px,92vw)] flex-col overflow-hidden rounded-xl border border-border bg-surface shadow-2xl">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-border px-4 py-2">
          <span className="text-sm font-medium text-zinc-200">Control Space</span>

          <div className="flex items-center gap-1">
            <button onClick={addCharacter} disabled={busy} className={btn}>
              Add character
            </button>
            <span className="px-1 text-xs text-zinc-400">
              {active + 1}/{characters.length}
            </span>
            <button onClick={() => turn(Math.PI)} disabled={busy} className={btn}>
              Turn around
            </button>
            <button
              onClick={() => turn(-Math.PI / 4)}
              disabled={busy}
              className={btn}
              title="Rotate left"
            >
              <RotateIcon />
            </button>
            <button
              onClick={() => turn(Math.PI / 4)}
              disabled={busy}
              className={btn}
              title="Rotate right"
            >
              <RotateIcon cw />
            </button>
            <button
              onClick={removeActive}
              disabled={busy || characters.length <= 1}
              className={btn}
            >
              Remove
            </button>
          </div>

          <div className="flex items-center gap-1">
            {presetNames.map((name) => (
              <button
                key={name}
                onClick={() => applyPreset(name)}
                disabled={busy}
                className={`${btn} capitalize`}
              >
                {name}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-1">
            <button
              onClick={frameActive}
              disabled={busy}
              className={btn}
              title="Frame the active figure"
            >
              Frame
            </button>
            <button onClick={resetView} disabled={busy} className={btn} title="Reset the camera">
              Reset view
            </button>
          </div>

          <div className="flex items-center overflow-hidden rounded-md border border-border">
            {(['pose', 'depth'] as const).map((k) => (
              <button
                key={k}
                onClick={() => setMapKind(k)}
                disabled={busy}
                className={`px-2 py-1 text-xs capitalize disabled:opacity-50 ${
                  mapKind === k ? 'bg-panel text-white' : 'text-zinc-300 hover:bg-panel/60'
                }`}
              >
                {k}
              </button>
            ))}
          </div>

          <div
            className="flex items-center overflow-hidden rounded-md border border-border"
            title="Output aspect - match it to the gen node's width/height so the pose isn't stretched"
          >
            {ASPECTS.map((a) => (
              <button
                key={a.label}
                onClick={() => setAspect(a.value)}
                disabled={busy}
                className={`px-2 py-1 text-xs disabled:opacity-50 ${
                  Math.abs(aspect - a.value) < 0.001
                    ? 'bg-panel text-white'
                    : 'text-zinc-300 hover:bg-panel/60'
                }`}
              >
                {a.label}
              </button>
            ))}
          </div>

          <label className="flex items-center gap-1.5 text-xs text-zinc-400">
            FOV
            <input
              type="range"
              min={20}
              max={90}
              value={fov}
              onChange={(e) => setFov(Number(e.target.value))}
              className="h-1 w-24 accent-emerald-500"
            />
          </label>

          <div className="ml-auto flex items-center gap-1">
            <button onClick={() => render('preview')} disabled={busy} className={btn}>
              Preview map
            </button>
            <button
              onClick={() => render('save')}
              disabled={busy}
              className="rounded-md border border-emerald-700 bg-emerald-600/20 px-2.5 py-1 text-xs text-emerald-200 hover:bg-emerald-600/30 disabled:opacity-50"
            >
              {busy ? 'Saving…' : 'Save to node'}
            </button>
            <button onClick={close} disabled={busy} className={btn}>
              Close
            </button>
          </div>
        </div>

        <div className="relative min-h-0 flex-1 bg-black">
          <Canvas camera={{ position: initialCamPos, fov }} className="h-full w-full">
            <color attach="background" args={['#0a0a0a']} />
            <ambientLight intensity={0.8} />
            <directionalLight position={[3, 5, 2]} intensity={0.6} />
            <Grid args={[10, 10]} cellColor="#333" sectionColor="#444" infiniteGrid />
            {characters.map((c, idx) => (
              <CharacterRig
                key={c.id}
                joints={c.joints}
                gizmoJoint={idx === active ? selJoint : null}
                dim={idx !== active}
                onPickJoint={(j) => {
                  setActive(idx)
                  setSelJoint(j)
                }}
                onMoveJoints={(joints) =>
                  setCharacters((cs) => cs.map((cc, i) => (i === idx ? { ...cc, joints } : cc)))
                }
                setOrbit={setOrbit}
              />
            ))}
            <OrbitControls
              ref={orbitRef}
              makeDefault
              enabled={orbit}
              target={initialTarget}
              onChange={captureCam}
            />
            <CameraRig fov={fov} />
            <PoseRenderer
              characters={characters.map((c) => c.joints)}
              nonce={nonce}
              enabled={mapKind === 'pose'}
              aspect={aspect}
              onRendered={handleRendered}
            />
            <DepthRenderer
              characters={characters.map((c) => c.joints)}
              nonce={nonce}
              enabled={mapKind === 'depth'}
              aspect={aspect}
              onRendered={handleRendered}
            />
          </Canvas>

          {/* Safe-frame guide: the region captured at the chosen output aspect (shares the camera's
              vertical view). Pose only what's inside it. */}
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
            <div
              className="h-full border border-dashed border-white/25 shadow-[0_0_0_9999px_rgba(0,0,0,0.35)]"
              style={{ aspectRatio: String(aspect) }}
            />
          </div>

          {preview && (
            <div className="absolute bottom-3 right-3 w-40 overflow-hidden rounded-md border border-border bg-black shadow-lg">
              <div className="border-b border-border px-2 py-1 text-[10px] capitalize text-zinc-400">
                {mapKind} control map
              </div>
              <img src={preview} alt={`${mapKind} control map`} className="block h-auto w-full" />
            </div>
          )}

          <div className="pointer-events-none absolute left-3 top-3 rounded-md bg-black/60 px-2.5 py-1.5 text-[11px] text-zinc-300">
            Click a joint to select, drag the gizmo or use arrow keys (Shift = depth) to pose.
            Orbit/zoom with the mouse.
          </div>

          {/* Which way the camera sees the character, and the prompt text that facing adds. */}
          <div className="absolute right-3 top-3 max-w-[15rem] rounded-md bg-black/60 px-2.5 py-1.5 text-[11px]">
            <div className="flex items-center gap-1.5">
              <span className="text-zinc-400">Figure {active + 1}:</span>
              <span
                className={
                  activeFacing === 'back'
                    ? 'font-medium text-amber-300'
                    : 'font-medium text-zinc-200'
                }
              >
                {activeFacing ? FACING_TEXT[activeFacing] : 'unknown'}
              </span>
            </div>
            <label className="mt-1 flex cursor-pointer items-start gap-1.5 text-zinc-400">
              <input
                type="checkbox"
                checked={applyHint}
                onChange={(e) => setApplyHint(e.target.checked)}
                className="mt-0.5 accent-emerald-500"
              />
              <span>
                Add facing to prompt
                {hint && <span className="block text-zinc-500">“{hint.positive}”</span>}
                {applyHint && !hint && (
                  <span className="block text-zinc-500">
                    {facings.length > 1 && new Set(facings).size > 1
                      ? 'mixed facings - nothing added'
                      : 'nothing needed for this facing'}
                  </span>
                )}
              </span>
            </label>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  )
}
