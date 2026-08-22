import type { ModelMeta } from '../../api/schemas'

interface CalibrationPoint {
  predicted: number
  actual: number
  count: number
}

interface Calibration {
  points: CalibrationPoint[]
  error: number
  bias: number
}

const SIZE = 220
const PAD = 28
const WELL_CALIBRATED = 0.05
const ROUGH = 0.1

function calibrationFrom(meta: ModelMeta): Calibration | null {
  const raw = meta.metrics?.calibration as Calibration | undefined
  return raw && Array.isArray(raw.points) && raw.points.length > 1 ? raw : null
}

function verdict(calibration: Calibration): { text: string; tone: 'good' | 'warn' } {
  if (calibration.error < WELL_CALIBRATED) {
    return {
      text: 'Well calibrated — when the model says 70%, it happens about 70% of the time. You can read its percentages at face value.',
      tone: 'good',
    }
  }
  const direction =
    calibration.bias > 0
      ? 'overconfident: real outcomes happen less often than its percentages suggest'
      : 'underconfident: real outcomes happen more often than its percentages suggest'
  if (calibration.error < ROUGH) {
    return { text: `Slightly ${direction}. Close enough for most uses.`, tone: 'good' }
  }
  return {
    text: `Noticeably ${direction}. Treat the percentages as a ranking (higher = more likely), not a promise.`,
    tone: 'warn',
  }
}

/** Reliability plot: predicted probability vs how often it actually happened. */
export function CalibrationCard({ meta }: { meta: ModelMeta }) {
  const calibration = calibrationFrom(meta)
  if (!calibration) return null

  const classes = (meta.metrics?.classes as string[] | undefined) ?? []
  const positive = classes[1] ?? 'yes'
  const plot = SIZE - 2 * PAD
  const x = (v: number) => PAD + v * plot
  const y = (v: number) => SIZE - PAD - v * plot
  const { text, tone } = verdict(calibration)

  return (
    <div className="card" style={{ marginBottom: '1.25rem' }}>
      <h2>Can you trust the percentages?</h2>
      <p className="muted small">
        Each dot groups held-out rows by the model’s stated confidence in “{positive}” —
        on the dashed line, stated confidence matches reality.
      </p>
      <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
        <svg viewBox={`0 0 ${SIZE} ${SIZE}`} width={SIZE} role="img">
          <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="var(--border)" strokeDasharray="4 3" />
          <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(0)} stroke="var(--border)" />
          <line x1={x(0)} y1={y(0)} x2={x(0)} y2={y(1)} stroke="var(--border)" />
          {calibration.points.map((p, i) => (
            <circle
              key={i}
              cx={x(p.predicted)}
              cy={y(p.actual)}
              r={4.5}
              fill="var(--primary)"
              stroke="var(--surface)"
              strokeWidth={1.5}
            >
              <title>{`Model said ${Math.round(p.predicted * 100)}% → happened ${Math.round(p.actual * 100)}% of the time (${p.count} rows)`}</title>
            </circle>
          ))}
          <text x={x(0.5)} y={SIZE - 6} textAnchor="middle" fontSize={10} fill="var(--text-muted)">
            model’s stated confidence
          </text>
          <text
            x={8}
            y={y(0.5)}
            fontSize={10}
            fill="var(--text-muted)"
            transform={`rotate(-90 8 ${y(0.5)})`}
            textAnchor="middle"
          >
            how often it happened
          </text>
        </svg>
        <p
          className="small"
          style={{
            flex: 1,
            minWidth: 180,
            color: tone === 'good' ? 'var(--success)' : 'var(--warning-text)',
          }}
        >
          {text}
        </p>
      </div>
    </div>
  )
}
