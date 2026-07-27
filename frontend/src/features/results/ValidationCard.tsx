import { useValidationRows } from '../../api/hooks'
import type { ModelMeta } from '../../api/schemas'
import { formatCell } from '../../lib/formatters'

export function ValidationCard({ meta }: { meta: ModelMeta }) {
  const validation = useValidationRows(meta.id, meta.status === 'complete')

  if (validation.error) return null // older models have no saved validation
  if (!validation.data) return null

  const { columns, rows, total_rows } = validation.data
  const isClassification = meta.task === 'classification'

  return (
    <div className="card" style={{ marginTop: '1.25rem' }}>
      <div className="page-title-row">
        <div>
          <h2>Proof on rows the model never saw</h2>
          <p className="muted small">
            These {total_rows.toLocaleString()} rows were held back from training — the
            model predicted them blind. Compare <strong>{meta.target_column}</strong> with
            the model’s <strong>predicted</strong> value.
          </p>
        </div>
        <a className="btn" href={`/api/models/${meta.id}/validation/download`} download>
          Download all {total_rows.toLocaleString()} rows (CSV)
        </a>
      </div>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((column) => (
                <th
                  key={column}
                  className={
                    column === 'predicted' || column === meta.target_column
                      ? 'validation-highlight'
                      : undefined
                  }
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={index}>
                {columns.map((column) => (
                  <td
                    key={column}
                    className={
                      column === 'predicted' || column === meta.target_column
                        ? 'validation-highlight'
                        : undefined
                    }
                  >
                    {column === 'correct' && isClassification ? (
                      <span className={row[column] ? 'tick' : 'cross'}>
                        {row[column] ? '✓' : '✗'}
                      </span>
                    ) : (
                      formatCell(row[column])
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
