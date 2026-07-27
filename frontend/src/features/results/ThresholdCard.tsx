import { useState } from 'react'

import type { ModelMeta, ThresholdPoint } from '../../api/schemas'
import { formatPercent } from '../../lib/formatters'

function curveFrom(meta: ModelMeta): ThresholdPoint[] | null {
  const raw = meta.metrics?.threshold_curve
  return Array.isArray(raw) ? (raw as ThresholdPoint[]) : null
}

export function ThresholdCard({ meta }: { meta: ModelMeta }) {
  const [threshold, setThreshold] = useState(0.5)
  const curve = curveFrom(meta)
  if (!curve) return null

  const classes = (meta.metrics?.classes as string[] | undefined) ?? []
  const positive = classes[1] ?? 'yes'
  const point =
    curve.reduce((best, p) =>
      Math.abs(p.threshold - threshold) < Math.abs(best.threshold - threshold) ? p : best,
    )

  return (
    <div className="card" style={{ marginBottom: '1.25rem' }}>
      <h2>Tune the decision cut-off</h2>
      <p className="muted small">
        The model flags a row as “{positive}” when its confidence passes this bar. Slide
        it to trade off catching more vs. fewer false alarms — measured on the held-back
        test rows.
      </p>
      <div className="threshold-slider-row">
        <span className="muted small">flag more</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={threshold}
          aria-label="Decision threshold"
          onChange={(event) => setThreshold(Number(event.target.value))}
        />
        <span className="muted small">flag less</span>
        <span className="threshold-value">{formatPercent(point.threshold)}</span>
      </div>
      <div className="threshold-stats">
        <div>
          <strong>{formatPercent(point.recall)}</strong>
          <span className="muted small"> of all true “{positive}” rows get caught</span>
        </div>
        <div>
          <strong>{point.precision !== null ? formatPercent(point.precision) : '—'}</strong>
          <span className="muted small"> of flagged rows really are “{positive}”</span>
        </div>
        <div>
          <strong>{point.flagged_pct}%</strong>
          <span className="muted small"> of rows get flagged</span>
        </div>
        <div>
          <strong>{formatPercent(point.accuracy)}</strong>
          <span className="muted small"> overall accuracy at this cut-off</span>
        </div>
      </div>
    </div>
  )
}
