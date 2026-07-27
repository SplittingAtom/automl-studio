import type { ColumnProfile } from '../../api/schemas'
import { columnKindLabel } from '../../lib/formatters'

const SELECTABLE_KINDS = new Set(['numeric', 'categorical'])

export function TargetPicker({
  columns,
  selected,
  onSelect,
}: {
  columns: ColumnProfile[]
  selected: string | null
  onSelect: (name: string) => void
}) {
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            <th aria-label="Select" />
            <th>Column</th>
            <th>Type</th>
            <th>Missing</th>
            <th>Unique values</th>
          </tr>
        </thead>
        <tbody>
          {columns.map((column) => {
            const selectable = SELECTABLE_KINDS.has(column.kind)
            const classes = [
              'column-row',
              selectable ? '' : 'disabled',
              selected === column.name ? 'selected' : '',
            ]
              .filter(Boolean)
              .join(' ')
            return (
              <tr
                key={column.name}
                className={classes}
                onClick={() => selectable && onSelect(column.name)}
                title={selectable ? undefined : 'This column can’t be predicted.'}
              >
                <td>
                  <input
                    type="radio"
                    name="target"
                    aria-label={`Predict ${column.name}`}
                    checked={selected === column.name}
                    disabled={!selectable}
                    onChange={() => onSelect(column.name)}
                  />
                </td>
                <td>
                  <strong>{column.name}</strong>
                </td>
                <td>
                  <span className={`chip chip-${selectable ? column.kind : 'excluded'}`}>
                    {columnKindLabel(column.kind)}
                  </span>
                </td>
                <td>{column.missing_pct > 0 ? `${column.missing_pct}%` : '—'}</td>
                <td>{column.unique_count.toLocaleString()}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
