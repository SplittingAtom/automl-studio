import type { ColumnExploration } from '../../api/schemas'
import { columnKindLabel, formatNumber } from '../../lib/formatters'
import { CategoryBars, Histogram } from './DistributionCharts'

const HIGH_MISSING_WARN_PCT = 20

/** One dataset column: kind, distribution chart, stats, and caveats. */
export function ColumnCard({
  column,
  totalRows,
}: {
  column: ColumnExploration
  totalRows: number
}) {
  return (
    <div className="card column-card">
      <div className="column-card-header">
        <h3 title={column.name}>{column.name}</h3>
        <span className={`chip chip-${chipKind(column.kind)}`} title={kindTip(column.kind)}>
          {columnKindLabel(column.kind)}
        </span>
      </div>
      <p
        className="column-card-meta muted small"
        title="Unique = distinct values · missing = share of empty cells · outliers = values far outside the typical range (beyond 1.5× the middle spread)"
      >
        {column.unique_count.toLocaleString()} unique
        {column.missing_pct > 0 && (
          <>
            {' · '}
            <span className={column.missing_pct > HIGH_MISSING_WARN_PCT ? 'missing-high' : ''}>
              {column.missing_pct}% missing
            </span>
          </>
        )}
        {column.stats !== null && column.stats.outlier_count > 0 && (
          <> · {column.stats.outlier_count.toLocaleString()} outliers</>
        )}
      </p>
      <Chart column={column} totalRows={totalRows} />
      {column.stats !== null && (
        <p
          className="column-card-stats muted small"
          title="Median = the middle value (half the rows are below it) · mean = the average · spread = how far values typically sit from the average (one standard deviation)"
        >
          min {formatNumber(column.stats.min)} · median {formatNumber(column.stats.median)}
          {' '}· mean {formatNumber(column.stats.mean)} · max {formatNumber(column.stats.max)}
          {' '}· spread ±{formatNumber(column.stats.std)}
        </p>
      )}
      {column.note !== null && <p className="muted small column-card-note">{column.note}</p>}
    </div>
  )
}

function Chart({ column, totalRows }: { column: ColumnExploration; totalRows: number }) {
  if (column.kind === 'numeric') {
    return <Histogram bins={column.bins} totalRows={totalRows} />
  }
  if (column.kind === 'categorical') {
    return (
      <CategoryBars
        bins={column.bins}
        otherCount={column.other_count}
        totalRows={totalRows}
      />
    )
  }
  if (column.kind === 'datetime') {
    return <Histogram bins={column.bins} totalRows={totalRows} timeAxis />
  }
  return null // id_like / unsupported: the note says why there's no chart
}

function kindTip(kind: ColumnExploration['kind']): string {
  const tips: Record<ColumnExploration['kind'], string> = {
    numeric: 'A number column: gets a histogram here and a slider in what-if scenarios.',
    categorical: 'A category column: a limited set of values. Gets a dropdown in what-if scenarios.',
    datetime: 'A date column: models use its year, month, and day of week, and it can order rows for time-aware training.',
    id_like: 'Looks like a row identifier — unique labels carry no pattern, so models leave it out.',
    unsupported: 'Completely empty — nothing to learn from.',
  }
  return tips[kind]
}

function chipKind(kind: ColumnExploration['kind']): string {
  if (kind === 'id_like' || kind === 'unsupported') return 'excluded'
  return kind
}
