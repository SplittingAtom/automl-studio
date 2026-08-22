import type { DistributionBin } from '../../api/schemas'
import { formatNumber } from '../../lib/formatters'

const WIDTH = 280
const PLOT_HEIGHT = 78
const LABEL_BAND = 16
const BAR_GAP = 2
const MAX_CATEGORY_ROWS = 9 // 8 top categories + Other

/** Vertical mini-histogram for numeric and datetime columns. */
export function Histogram({
  bins,
  totalRows,
  timeAxis = false,
}: {
  bins: DistributionBin[]
  totalRows: number
  timeAxis?: boolean
}) {
  if (bins.length === 0) return <p className="muted small">No values to plot.</p>
  const maxCount = Math.max(...bins.map((b) => b.count), 1)
  const barWidth = (WIDTH - BAR_GAP * (bins.length - 1)) / bins.length
  const leftLabel = timeAxis ? bins[0].label : formatEdge(bins[0].low)
  const lastBin = bins[bins.length - 1]
  const rightLabel = timeAxis ? lastBin.label : formatEdge(lastBin.high)

  return (
    <svg viewBox={`0 0 ${WIDTH} ${PLOT_HEIGHT + LABEL_BAND}`} width="100%" role="img">
      {bins.map((bin, i) => {
        const height =
          bin.count === 0 ? 0 : Math.max(2, (bin.count / maxCount) * PLOT_HEIGHT)
        return (
          <rect
            key={i}
            x={i * (barWidth + BAR_GAP)}
            y={PLOT_HEIGHT - height}
            width={barWidth}
            height={height}
            rx={Math.min(1.5, barWidth / 2)}
            fill="var(--primary)"
          >
            <title>
              {`${bin.label}: ${bin.count} rows (${percent(bin.count, totalRows)})`}
            </title>
          </rect>
        )
      })}
      <line
        x1={0}
        x2={WIDTH}
        y1={PLOT_HEIGHT}
        y2={PLOT_HEIGHT}
        stroke="var(--border)"
        strokeWidth={1}
      />
      <text x={0} y={PLOT_HEIGHT + 12} fontSize={10} fill="var(--text-muted)">
        {leftLabel}
      </text>
      <text
        x={WIDTH}
        y={PLOT_HEIGHT + 12}
        textAnchor="end"
        fontSize={10}
        fill="var(--text-muted)"
      >
        {rightLabel}
      </text>
    </svg>
  )
}

/** Horizontal top-categories bars for categorical columns. */
export function CategoryBars({
  bins,
  otherCount,
  totalRows,
}: {
  bins: DistributionBin[]
  otherCount: number
  totalRows: number
}) {
  if (bins.length === 0) return <p className="muted small">No values to plot.</p>
  // The folded tail is flagged, not identified by label — a real category can
  // legitimately be named "Other".
  const rows = bins.map((bin) => ({ ...bin, folded: false }))
  if (otherCount > 0) {
    rows.push({
      label: 'All others',
      count: otherCount,
      low: null,
      high: null,
      folded: true,
    })
  }
  const maxCount = Math.max(...rows.map((r) => r.count), 1)

  return (
    <div className="cat-bars">
      {rows.slice(0, MAX_CATEGORY_ROWS).map((row, i) => (
        <div
          key={i}
          className="cat-bar-row"
          title={`${row.label}: ${row.count} rows (${percent(row.count, totalRows)})`}
        >
          <span className="cat-bar-label">{row.label}</span>
          <div className="cat-bar-track">
            <div
              className="cat-bar-fill"
              style={{
                width: `${(row.count / maxCount) * 100}%`,
                background: row.folded ? 'var(--text-muted)' : 'var(--primary)',
              }}
            />
          </div>
          <span className="cat-bar-count">{row.count.toLocaleString()}</span>
        </div>
      ))}
    </div>
  )
}

function formatEdge(value: number | null): string {
  return value === null ? '' : formatNumber(value)
}

function percent(count: number, total: number): string {
  if (total === 0) return '0%'
  const share = (100 * count) / total
  return share < 1 && count > 0 ? '<1%' : `${Math.round(share)}%`
}
