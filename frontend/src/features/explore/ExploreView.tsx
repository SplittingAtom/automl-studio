import { useDatasetExploration } from '../../api/hooks'
import { ErrorBanner } from '../../components/ErrorBanner'
import { ColumnCard } from './ColumnCard'

/**
 * The data profiling view: dataset-level overview tiles plus one
 * distribution card per column.
 */
export function ExploreView({ datasetId }: { datasetId: string }) {
  const exploration = useDatasetExploration(datasetId)

  if (exploration.isLoading) {
    return <p className="muted">Profiling the data…</p>
  }
  if (exploration.error || !exploration.data) {
    return <ErrorBanner error={exploration.error} />
  }

  const data = exploration.data
  const tiles = [
    {
      value: data.row_count.toLocaleString(),
      label: 'rows',
      tip: 'Total rows in the dataset. Models need at least 50 usable rows; hundreds or more give steadier results.',
    },
    {
      value: String(data.column_count),
      label: 'columns',
      tip: 'Total columns, including any the models will skip (IDs, empty columns).',
    },
    {
      value: `${data.missing_cells_pct}%`,
      label: 'of all cells are empty',
      tip: 'Share of empty cells across the whole table. Models handle missing values automatically, but columns that are mostly empty are left out.',
    },
    {
      value: data.duplicate_rows.toLocaleString(),
      label: data.duplicate_rows === 1 ? 'exact duplicate row' : 'exact duplicate rows',
      tip: 'Rows identical to another row in every column. A few can be legitimate; many usually mean the same records were loaded twice.',
    },
  ]

  return (
    <div>
      <div className="metric-cards">
        {tiles.map((tile) => (
          <div className="metric-card" key={tile.label} title={tile.tip}>
            <div className="value">{tile.value}</div>
            <div className="label">{tile.label}</div>
          </div>
        ))}
      </div>
      {data.highlights.length > 0 && (
        <div className="card" style={{ marginTop: '1.25rem' }}>
          <h2>Worth a look</h2>
          <ul className="highlight-list">
            {data.highlights.map((highlight) => (
              <li key={highlight.message} className={`highlight-${highlight.tone}`}>
                {highlight.message}
              </li>
            ))}
          </ul>
        </div>
      )}
      <div className="column-grid">
        {data.columns.map((column) => (
          <ColumnCard key={column.name} column={column} totalRows={data.row_count} />
        ))}
      </div>
    </div>
  )
}
