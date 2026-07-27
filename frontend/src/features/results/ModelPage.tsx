import { Link, useParams } from 'react-router-dom'

import { useModel } from '../../api/hooks'
import { ErrorBanner } from '../../components/ErrorBanner'
import { WarningBanner } from '../../components/WarningBanner'
import { taskLabel } from '../../lib/formatters'
import { WhatIfPanel } from '../whatif/WhatIfPanel'
import { ImportanceChart } from './ImportanceChart'
import { MetricsCards } from './MetricsCards'
import { TrainingProgress } from './TrainingProgress'

export function ModelPage() {
  const { id = '' } = useParams()
  const model = useModel(id)

  if (model.isLoading) return <p className="muted">Loading model…</p>
  if (model.error || !model.data) return <ErrorBanner error={model.error} />

  const meta = model.data

  if (meta.status === 'queued' || meta.status === 'training') {
    return <TrainingProgress status={meta.status} />
  }

  if (meta.status === 'failed') {
    return (
      <div className="card" style={{ maxWidth: 560, margin: '3rem auto', textAlign: 'center' }}>
        <h2>Training didn’t work out</h2>
        <p className="muted">{meta.error ?? 'Something went wrong during training.'}</p>
        <Link className="btn" to={`/datasets/${meta.dataset_id}/configure`}>
          Try different settings
        </Link>
      </div>
    )
  }

  return (
    <div>
      <h1>
        Predicting <em>{meta.target_column}</em> — {meta.dataset_name}
      </h1>
      <p className="muted">
        {taskLabel(meta.task)} · trained on {meta.n_rows_used?.toLocaleString()} rows
      </p>
      <WarningBanner warnings={meta.warnings} />

      <div className="results-grid">
        <div>
          <MetricsCards meta={meta} />
          <div className="card">
            <h2>What drives the predictions?</h2>
            <p className="muted small">
              The share of the model’s decisions each column is responsible for.
            </p>
            <ImportanceChart importance={meta.importance ?? []} />
            {meta.excluded_columns.length > 0 && (
              <p className="muted small" style={{ marginBottom: 0 }}>
                Not used: {meta.excluded_columns.map((c) => c.name).join(', ')}
              </p>
            )}
          </div>
        </div>
        <WhatIfPanel model={meta} />
      </div>
    </div>
  )
}
