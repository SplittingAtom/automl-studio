import type { InsightsResponse } from '../../api/schemas'
import { NEUTRAL_DOT, dotColor, jitter } from './color'

const WIDTH = 680
const LABEL_WIDTH = 140
const ROW_HEIGHT = 36
const AXIS_BAND = 34
const DOT_RADIUS = 4.5
const RIGHT_PAD = 16

/**
 * SHAP summary ("beeswarm"): one dot per held-out row per feature. Horizontal
 * position is how far that row's value pushed the prediction; color is the
 * value itself (light = low, dark = high, gray = category).
 */
export function ImpactBeeswarm({ insights }: { insights: InsightsResponse }) {
  const impacts = insights.impacts
  if (impacts.length === 0) {
    return <p className="muted">No impact data available.</p>
  }

  const maxAbs = Math.max(
    0.001,
    ...impacts.flatMap((f) => f.points.map((p) => Math.abs(p.contribution))),
  )
  const plotWidth = WIDTH - LABEL_WIDTH - RIGHT_PAD
  const x = (contribution: number) =>
    LABEL_WIDTH + ((contribution + maxAbs) / (2 * maxAbs)) * plotWidth
  const height = impacts.length * ROW_HEIGHT + AXIS_BAND
  const zeroX = x(0)

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${height}`}
      width="100%"
      role="img"
      aria-label="How each column's values push predictions up or down"
    >
      {impacts.map((impact, row) => {
        const centerY = row * ROW_HEIGHT + ROW_HEIGHT / 2
        return (
          <g key={impact.feature}>
            {row > 0 && (
              <line
                x1={LABEL_WIDTH}
                x2={WIDTH - RIGHT_PAD}
                y1={row * ROW_HEIGHT}
                y2={row * ROW_HEIGHT}
                stroke="var(--border)"
                strokeWidth={1}
              />
            )}
            <text
              x={LABEL_WIDTH - 10}
              y={centerY}
              textAnchor="end"
              dominantBaseline="middle"
              fontSize={12}
              fill="var(--text)"
            >
              {truncate(impact.feature)}
            </text>
            {impact.points.map((point, i) => (
              <circle
                key={i}
                cx={x(point.contribution)}
                cy={centerY + jitter(i) * (ROW_HEIGHT - 2 * DOT_RADIUS - 4)}
                r={DOT_RADIUS}
                fill={dotColor(point.value_norm)}
                stroke="var(--surface)"
                strokeWidth={1.5}
              >
                <title>
                  {`${impact.feature} = ${point.value_label} → ${
                    point.contribution >= 0 ? '+' : ''
                  }${point.contribution} (${
                    point.contribution >= 0
                      ? insights.axis_high_label
                      : insights.axis_low_label
                  })`}
                </title>
              </circle>
            ))}
          </g>
        )
      })}
      <line
        x1={zeroX}
        x2={zeroX}
        y1={0}
        y2={impacts.length * ROW_HEIGHT}
        stroke="var(--border)"
        strokeWidth={1}
      />
      <text
        x={LABEL_WIDTH}
        y={impacts.length * ROW_HEIGHT + 20}
        fontSize={11}
        fill="var(--text-muted)"
      >
        ← {insights.axis_low_label}
      </text>
      <text
        x={WIDTH - RIGHT_PAD}
        y={impacts.length * ROW_HEIGHT + 20}
        textAnchor="end"
        fontSize={11}
        fill="var(--text-muted)"
      >
        {insights.axis_high_label} →
      </text>
    </svg>
  )
}

function truncate(name: string, max = 18): string {
  return name.length > max ? `${name.slice(0, max - 1)}…` : name
}

/** Legend row rendered in HTML above the chart. */
export function BeeswarmLegend() {
  return (
    <div className="beeswarm-legend">
      <span className="muted small">column value:</span>
      <span className="muted small">low</span>
      <div
        className="legend-gradient"
        style={{
          background: `linear-gradient(to right, ${dotColor(0)}, ${dotColor(0.5)}, ${dotColor(1)})`,
        }}
      />
      <span className="muted small">high</span>
      <span className="legend-dot" style={{ background: NEUTRAL_DOT }} />
      <span className="muted small">category or missing</span>
    </div>
  )
}
