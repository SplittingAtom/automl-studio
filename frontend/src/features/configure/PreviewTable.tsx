import type { DatasetPreview } from '../../api/schemas'
import { ErrorBanner } from '../../components/ErrorBanner'
import { formatCell } from '../../lib/formatters'

export function PreviewTable({
  preview,
  loading,
  error,
}: {
  preview: DatasetPreview | undefined
  loading: boolean
  error: unknown
}) {
  if (loading) return <p className="muted">Loading preview…</p>
  if (error) return <ErrorBanner error={error} />
  if (!preview) return null

  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            {preview.columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {preview.rows.map((row, index) => (
            <tr key={index}>
              {preview.columns.map((column) => (
                <td key={column}>{formatCell(row[column])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
