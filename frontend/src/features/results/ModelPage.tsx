import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { useModel } from '../../api/hooks'
import { ErrorBanner } from '../../components/ErrorBanner'
import { WarningBanner } from '../../components/WarningBanner'
import { taskLabel } from '../../lib/formatters'
import { ForecastView } from './ForecastView'
import { ResultsDashboard } from './ResultsDashboard'
import { RetrainSuggestions } from './RetrainSuggestions'
import { TrainingProgress } from './TrainingProgress'

type View = 'overview' | 'forecast'

export function ModelPage() {
  const { id = '' } = useParams()
  const model = useModel(id)
  const [view, setView] = useState<View>('overview')

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

  // The forecast chart needs a timeline and a numeric prediction
  const hasForecastView = meta.time_column !== null && meta.task === 'regression'

  return (
    <div>
      <div className="page-title-row">
        <div>
          <h1>
            Predicting <em>{meta.target_column}</em> — {meta.dataset_name}
          </h1>
          <p className="muted">
            {taskLabel(meta.task)} · trained on {meta.n_rows_used?.toLocaleString()} rows
            {meta.time_column && ` · time-aware, ordered by ${meta.time_column}`}
            {meta.horizon > 0 && ` (${meta.horizon}-row horizon)`}
          </p>
        </div>
        <div className="title-actions">
          <a
            className="btn"
            href={`/api/models/${meta.id}/export`}
            download
            title="Self-contained scoring kit: model + predict.py, runs anywhere with Python"
          >
            Download model
          </a>
          <Link className="btn" to={`/datasets/${meta.dataset_id}/models`}>
            Compare models
          </Link>
          <Link className="btn" to={`/datasets/${meta.dataset_id}/configure`}>
            Train another
          </Link>
        </div>
      </div>

      {hasForecastView && (
        <div className="tab-bar" role="tablist">
          <button
            role="tab"
            aria-selected={view === 'overview'}
            className={`tab${view === 'overview' ? ' active' : ''}`}
            onClick={() => setView('overview')}
          >
            Overview
          </button>
          <button
            role="tab"
            aria-selected={view === 'forecast'}
            className={`tab${view === 'forecast' ? ' active' : ''}`}
            onClick={() => setView('forecast')}
          >
            Time-series prediction
          </button>
        </div>
      )}

      {view === 'forecast' && hasForecastView ? (
        <ForecastView meta={meta} />
      ) : (
        <>
          <WarningBanner warnings={meta.warnings} />
          <RetrainSuggestions meta={meta} />
          <ResultsDashboard meta={meta} />
        </>
      )}
    </div>
  )
}
