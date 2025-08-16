import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Scene3D } from '../viz/Scene3D'
import { scaleLinear } from 'd3-scale'

type EnergyPoint = { key: string; label: string; value_pct: number; color: string }
type Payload = {
  year: number
  series: EnergyPoint[]
  kpis: { total_mwh?: number | null; pct_renewables: number; co2?: number | null; savings?: number | null }
  focus: string
  chapter: string
  meta: { title: string; unit: string; updated_at: number }
}

export const App: React.FC = () => {
  const [data, setData] = useState<Payload | null>(() => (window as any).__INITIAL_ENERGY_PAYLOAD__ || null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    function onMessage(ev: MessageEvent) {
      if (ev.data && ev.data.type === 'ENERGY_PAYLOAD') {
        setData(ev.data.payload)
      }
    }
    window.addEventListener('message', onMessage)
    window.parent && window.parent.postMessage && window.parent.postMessage({ type: 'FRONTEND_READY' }, '*')
    setReady(true)
    return () => window.removeEventListener('message', onMessage)
  }, [])

  const scaleHeight = useMemo(() => scaleLinear().domain([0, 100]).range([0.05, 2.4]).clamp(true), [])

  return (
    <div style={{ height: '100vh', width: '100vw', display: 'grid', gridTemplateRows: '48px 1fr', gap: 8 }}>
      <TopBar data={data} />
      <Scene3D payload={data} scaleHeight={scaleHeight} />
    </div>
  )
}

const TopBar: React.FC<{ data: Payload | null }> = ({ data }) => {
  return (
    <div style={{ display: 'flex', alignItems: 'center', padding: '8px 12px', gap: 16, borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
      <div style={{ fontWeight: 700, letterSpacing: 0.5 }}>EnergyScape 3D</div>
      <div style={{ opacity: 0.8 }}>Année: <b>{data?.year ?? '-'}</b></div>
      <div style={{ opacity: 0.8, marginLeft: 16 }}>Chapitre: <b>{data?.chapter ?? '-'}</b></div>
      <div style={{ marginLeft: 'auto', opacity: 0.7 }}>Focus: {data?.focus ?? 'none'}</div>
    </div>
  )
}


