import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { useModel } from '../../api/hooks'
import type { ModelMeta } from '../../api/schemas'
import { ErrorBanner } from '../../components/ErrorBanner'
import { TabBar } from '../../components/TabBar'
import { WarningBanner } from '../../components/WarningBanner'
import { taskLabel } from '../../lib/formatters'
import { InsightsView } from '../insights/InsightsView'
import { ForecastView } from './ForecastView'
import { ResultsDashboard } from './ResultsDashboard'
import { RetrainSuggestions } from './RetrainSuggestions'
import { TrainingProgress } from './TrainingProgress'

type View = 'overview' | 'insights' | 'forecast'

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
          {meta.baseline_model_id && <BaselineDelta meta={meta} />}
        </div>
        <div className="title-actions">
          <Link
            className="btn"
            to={`/datasets/${meta.dataset_id}/configure?baseline=${meta.id}`}
            title="Start a new run with this model's settings pre-filled, then adjust the advanced knobs"
          >
            Fine-tune
          </Link>
          <a
            className="btn"
            href={`/api/models/${meta.id}/report`}
            download
            title="Shareable one-page HTML report: scores, drivers, caveats, settings"
          >
            Download report
          </a>
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

      <TabBar<View>
        tabs={[
          {
            id: 'overview',
            label: 'Overview',
            tip: 'Scores, decision tuning, what-if scenarios, and the held-out proof table.',
          },
          {
            id: 'insights',
            label: 'Column insights',
            tip: 'Global explainability: how column values push predictions, where the model struggles, a simplified flowchart, and how columns relate.',
          },
          ...(hasForecastView
            ? [
                {
                  id: 'forecast' as const,
                  label: 'Time-series prediction',
                  tip: 'Actual vs predicted over the held-out recent window, plus forecasting ahead.',
                },
              ]
            : []),
        ]}
        current={view}
        onSelect={setView}
      />

      {view === 'insights' ? (
        <InsightsView meta={meta} />
      ) : view === 'forecast' && hasForecastView ? (
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

/** Headline-metric comparison against the run this one was tuned from. */
function BaselineDelta({ meta }: { meta: ModelMeta }) {
  const baseline = useModel(meta.baseline_model_id ?? '', meta.baseline_model_id !== null)
  const from = baseline.data
  if (!from?.metrics || !meta.metrics) return null
  const key = meta.task === 'classification' ? 'accuracy' : 'r2'
  const label = meta.task === 'classification' ? 'accuracy' : 'R²'
  const tuned = meta.metrics[key]
  const base = from.metrics[key]
  if (typeof tuned !== 'number' || typeof base !== 'number') return null
  const diff = tuned - base
  const color = diff >= 0 ? 'var(--success)' : 'var(--danger)'
  return (
    <p className="small" style={{ margin: '0.15rem 0 0' }}>
      <Link to={`/models/${from.id}`}>vs baseline</Link>:{' '}
      <strong style={{ color }}>
        {diff >= 0 ? '+' : '−'}
        {Math.abs(diff).toFixed(4)} {label}
      </strong>{' '}
      ({base.toFixed(4)} → {tuned.toFixed(4)})
    </p>
  )
}
