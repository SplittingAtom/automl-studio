import { Link, useParams } from 'react-router-dom'

import { useModel } from '../../api/hooks'
import { ErrorBanner } from '../../components/ErrorBanner'
import { WarningBanner } from '../../components/WarningBanner'
import { taskLabel } from '../../lib/formatters'
import { ResultsDashboard } from './ResultsDashboard'
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
      <div className="page-title-row">
        <div>
          <h1>
            Predicting <em>{meta.target_column}</em> — {meta.dataset_name}
          </h1>
          <p className="muted">
            {taskLabel(meta.task)} · trained on {meta.n_rows_used?.toLocaleString()} rows
          </p>
        </div>
        <div className="title-actions">
          <Link className="btn" to={`/datasets/${meta.dataset_id}/models`}>
            Compare models
          </Link>
          <Link className="btn" to={`/datasets/${meta.dataset_id}/configure`}>
            Train another
          </Link>
        </div>
      </div>
      <WarningBanner warnings={meta.warnings} />
      <ResultsDashboard meta={meta} />
    </div>
  )
}
