import React, { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

type EnergyPoint = { key: string; label: string; value_pct: number; color: string }
type Payload = {
  year: number
  series: EnergyPoint[]
  kpis: { total_mwh?: number | null; pct_renewables: number; co2?: number | null; savings?: number | null }
  focus: string
  meta: { title: string; unit: string; updated_at: number }
}

type Props = {
  payload: Payload | null
  scaleHeight: (v: number) => number
}

export const Scene3D: React.FC<Props> = ({ payload, scaleHeight }) => {
  const mountRef = useRef<HTMLDivElement | null>(null)
  const rafRef = useRef<number | null>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const barsRef = useRef<Record<string, THREE.Mesh>>({})
  const controlsRef = useRef<OrbitControls | null>(null)

  useEffect(() => {
    if (!mountRef.current) return
    const width = mountRef.current.clientWidth
    const height = mountRef.current.clientHeight

    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#0b0f1a')
    sceneRef.current = scene

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100)
    camera.position.set(3.2, 2.8, 4.2)
    camera.lookAt(0, 0.8, 0)
    cameraRef.current = camera

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio))
    rendererRef.current = renderer
    mountRef.current.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.minDistance = 2
    controls.maxDistance = 12
    controls.maxPolarAngle = Math.PI * 0.49
    controls.target.set(0, 0.8, 0)
    controlsRef.current = controls

    const ambient = new THREE.AmbientLight('#8ed6ff', 0.28)
    scene.add(ambient)
    const dir = new THREE.DirectionalLight('#ffffff', 0.9)
    dir.position.set(4, 6, 4)
    scene.add(dir)

    const grid = new THREE.GridHelper(20, 20, 0x1a2433, 0x1a2433)
    ;(grid.material as THREE.Material).opacity = 0.22
    ;(grid.material as THREE.Material).transparent = true
    grid.position.y = -0.001
    scene.add(grid)

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(10, 10),
      new THREE.MeshStandardMaterial({ color: '#0e1424', metalness: 0.2, roughness: 0.8 })
    )
    floor.rotation.x = -Math.PI / 2
    scene.add(floor)

    function animate() {
      rafRef.current = requestAnimationFrame(animate)
      controls.update()
      renderer.render(scene, camera)
    }
    animate()

    function onResize() {
      if (!mountRef.current || !cameraRef.current || !rendererRef.current) return
      const w = mountRef.current.clientWidth
      const h = mountRef.current.clientHeight
      cameraRef.current.aspect = w / h
      cameraRef.current.updateProjectionMatrix()
      rendererRef.current.setSize(w, h)
    }
    const ro = new ResizeObserver(() => onResize())
    ro.observe(mountRef.current)

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      ro.disconnect()
      if (rendererRef.current) {
        mountRef.current?.removeChild(rendererRef.current.domElement)
        rendererRef.current.dispose()
      }
    }
  }, [])

  useEffect(() => {
    if (!payload || !sceneRef.current) return
    const scene = sceneRef.current

    const spacing = 1.2
    const keys = payload.series.map(s => s.key)
    const startX = -((keys.length - 1) * spacing) / 2

    for (let i = 0; i < payload.series.length; i++) {
      const s = payload.series[i]
      const x = startX + i * spacing

      let mesh = barsRef.current[s.key]
      const targetH = scaleHeight(s.value_pct)
      if (!mesh) {
        const geom = new THREE.BoxGeometry(0.6, Math.max(0.001, targetH), 0.6)
        const mat = new THREE.MeshStandardMaterial({ color: s.color, emissive: s.color, emissiveIntensity: 0.45, metalness: 0.3, roughness: 0.4 })
        mesh = new THREE.Mesh(geom, mat)
        mesh.position.set(x, targetH / 2, 0)
        barsRef.current[s.key] = mesh
        scene.add(mesh)
      } else {
        // simple tween-like approach
        const currentH = (mesh.geometry as THREE.BoxGeometry).parameters.height
        const newH = currentH + (targetH - currentH) * 0.2
        mesh.geometry.dispose()
        mesh.geometry = new THREE.BoxGeometry(0.6, Math.max(0.001, newH), 0.6)
        mesh.position.y = newH / 2
        ;(mesh.material as THREE.MeshStandardMaterial).color = new THREE.Color(s.color)
        ;(mesh.material as THREE.MeshStandardMaterial).emissive = new THREE.Color(s.color)
      }
    }

    // Remove bars not in payload
    Object.keys(barsRef.current).forEach(k => {
      if (!keys.includes(k)) {
        const m = barsRef.current[k]
        scene.remove(m)
        m.geometry.dispose()
        ;(m.material as THREE.Material).dispose()
        delete barsRef.current[k]
      }
    })
  }, [payload, scaleHeight])

  return <div ref={mountRef} style={{ width: '100%', height: '100%', borderRadius: 8, overflow: 'hidden' }} />
}


