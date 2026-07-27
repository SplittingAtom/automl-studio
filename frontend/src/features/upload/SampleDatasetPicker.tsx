import type { DatasetMeta } from '../../api/schemas'
import { ErrorBanner } from '../../components/ErrorBanner'

const SAMPLE_BLURBS: Record<string, string> = {
  ds_sample_titanic: 'Who survived the Titanic? Predict survival from age, ticket class, and more.',
  ds_sample_housing: 'What is a house worth? Predict California home values from neighborhood stats.',
}

export function SampleDatasetPicker({
  samples,
  loading,
  error,
  onPick,
}: {
  samples: DatasetMeta[]
  loading: boolean
  error: unknown
  onPick: (id: string) => void
}) {
  if (loading) return <p className="muted">Loading sample datasets…</p>
  if (error) return <ErrorBanner error={error} />
  if (samples.length === 0) return <p className="muted">No sample datasets available.</p>

  return (
    <div className="sample-grid">
      {samples.map((dataset) => (
        <button key={dataset.id} className="sample-card" onClick={() => onPick(dataset.id)}>
          <h3>{dataset.name}</h3>
          <p className="muted small">
            {SAMPLE_BLURBS[dataset.id] ?? 'A ready-to-use example dataset.'}
          </p>
          <span className="chip">
            {dataset.row_count.toLocaleString()} rows · {dataset.column_count} columns
          </span>
        </button>
      ))}
    </div>
  )
}
