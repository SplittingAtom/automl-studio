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
        <div title="Recall: of all the rows that truly are the positive case, how many does the model catch at this cut-off. Lower the bar to catch more (at the cost of false alarms).">
          <strong>{formatPercent(point.recall)}</strong>
          <span className="muted small"> of all true “{positive}” rows get caught</span>
        </div>
        <div title="Precision: of the rows the model flags, how many really are the positive case. Raise the bar to make each flag more reliable.">
          <strong>{point.precision !== null ? formatPercent(point.precision) : '—'}</strong>
          <span className="muted small"> of flagged rows really are “{positive}”</span>
        </div>
        <div title="Workload: the share of all rows that would be flagged at this cut-off — a proxy for how many cases you'd have to review.">
          <strong>{point.flagged_pct}%</strong>
          <span className="muted small"> of rows get flagged</span>
        </div>
        <div title="Overall accuracy if this cut-off were used for every decision.">
          <strong>{formatPercent(point.accuracy)}</strong>
          <span className="muted small"> overall accuracy at this cut-off</span>
        </div>
      </div>
    </div>
  )
}
