/**
 * One posable OpenPose character in the Control Space scene: joints as clickable spheres, bones as
 * colored lines. When this character is active, its selected joint carries a TransformControls move
 * gizmo; dragging it reports the new pose and pauses the orbit camera. Non-active characters render
 * dimmed so the active one reads clearly.
 */
import { useEffect, useRef } from 'react'
import { Line, TransformControls } from '@react-three/drei'
import * as THREE from 'three'
import { LIMBS, POSE_COLORS, type Vec3 } from './skeleton'

export function CharacterRig({
  joints,
  gizmoJoint,
  dim,
  onPickJoint,
  onMoveJoints,
  setOrbit,
}: {
  joints: Vec3[]
  /** Joint index to attach the move gizmo to, or null (only the active character passes a value). */
  gizmoJoint: number | null
  dim: boolean
  onPickJoint: (j: number) => void
  onMoveJoints: (joints: Vec3[]) => void
  setOrbit: (on: boolean) => void
}): React.JSX.Element {
  const meshes = useRef<(THREE.Mesh | null)[]>([])
  const transform = useRef<React.ComponentRef<typeof TransformControls>>(null)

  useEffect(() => {
    const controls = transform.current
    if (!controls) return
    const onDrag = (ev: { value: boolean }): void => setOrbit(!ev.value)
    // three-stdlib's TransformControls fires 'dragging-changed', which isn't in Object3DEventMap.
    const evt = controls as unknown as {
      addEventListener(e: 'dragging-changed', cb: (ev: { value: boolean }) => void): void
      removeEventListener(e: 'dragging-changed', cb: (ev: { value: boolean }) => void): void
    }
    evt.addEventListener('dragging-changed', onDrag)
    return () => evt.removeEventListener('dragging-changed', onDrag)
  }, [gizmoJoint, setOrbit])

  const syncFromGizmo = (): void => {
    if (gizmoJoint === null) return
    const mesh = meshes.current[gizmoJoint]
    if (!mesh) return
    const next = joints.slice()
    next[gizmoJoint] = [mesh.position.x, mesh.position.y, mesh.position.z]
    onMoveJoints(next)
  }

  return (
    <group>
      {LIMBS.map(([a, b], i) => (
        <Line
          key={`limb-${i}`}
          points={[joints[a], joints[b]] as [number, number, number][]}
          color={POSE_COLORS[i % POSE_COLORS.length]}
          lineWidth={dim ? 2 : 3}
          transparent
          opacity={dim ? 0.4 : 1}
        />
      ))}
      {joints.map((j, i) => (
        <mesh
          key={`joint-${i}`}
          ref={(m) => {
            meshes.current[i] = m
          }}
          position={j as [number, number, number]}
          onClick={(e) => {
            e.stopPropagation()
            onPickJoint(i)
          }}
        >
          <sphereGeometry args={[0.03, 16, 16]} />
          <meshBasicMaterial
            color={gizmoJoint === i ? 'white' : POSE_COLORS[i % POSE_COLORS.length]}
            transparent
            opacity={dim ? 0.45 : 1}
          />
        </mesh>
      ))}
      {gizmoJoint !== null && meshes.current[gizmoJoint] && (
        <TransformControls
          ref={transform}
          object={meshes.current[gizmoJoint] ?? undefined}
          mode="translate"
          size={0.6}
          onObjectChange={syncFromGizmo}
        />
      )}
    </group>
  )
}
