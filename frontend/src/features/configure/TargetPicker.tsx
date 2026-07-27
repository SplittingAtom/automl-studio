import type { ColumnProfile } from '../../api/schemas'
import { columnKindLabel } from '../../lib/formatters'

const SELECTABLE_KINDS = new Set(['numeric', 'categorical'])
const USABLE_KINDS = new Set(['numeric', 'categorical', 'datetime'])

export function TargetPicker({
  columns,
  selected,
  excluded,
  onSelect,
  onToggleUse,
}: {
  columns: ColumnProfile[]
  selected: string | null
  excluded: Set<string>
  onSelect: (name: string) => void
  onToggleUse: (name: string) => void
}) {
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            <th>Predict</th>
            <th>Column</th>
            <th>Type</th>
            <th>Missing</th>
            <th>Unique values</th>
            <th title="Untick to leave a column out of the model">Use as input</th>
          </tr>
        </thead>
        <tbody>
          {columns.map((column) => {
            const selectable = SELECTABLE_KINDS.has(column.kind)
            const usable = USABLE_KINDS.has(column.kind)
            const isTarget = selected === column.name
            const classes = [
              'column-row',
              selectable ? '' : 'disabled',
              isTarget ? 'selected' : '',
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
                    checked={isTarget}
                    disabled={!selectable}
                    onChange={() => onSelect(column.name)}
                  />
                </td>
                <td>
                  <strong>{column.name}</strong>
                </td>
                <td>
                  <span className={`chip chip-${usable ? column.kind : 'excluded'}`}>
                    {columnKindLabel(column.kind)}
                  </span>
                </td>
                <td>{column.missing_pct > 0 ? `${column.missing_pct}%` : '—'}</td>
                <td>{column.unique_count.toLocaleString()}</td>
                <td onClick={(event) => event.stopPropagation()}>
                  {!isTarget && usable && (
                    <input
                      type="checkbox"
                      aria-label={`Use ${column.name} as an input`}
                      checked={!excluded.has(column.name)}
                      onChange={() => onToggleUse(column.name)}
                    />
                  )}
                  {isTarget && <span className="muted small">target</span>}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
