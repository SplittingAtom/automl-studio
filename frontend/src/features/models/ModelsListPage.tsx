import { Link, useParams } from 'react-router-dom'

import { useDataset, useModels } from '../../api/hooks'
import type { ModelMeta } from '../../api/schemas'
import { ErrorBanner } from '../../components/ErrorBanner'
import { formatNumber, formatPercent, taskLabel } from '../../lib/formatters'
import { bestModelId, rankModels, variantLabel } from './leaderboard'

function keyMetric(model: ModelMeta): string {
  const metrics = model.metrics ?? {}
  if (model.task === 'classification' && typeof metrics.accuracy === 'number') {
    return `${formatPercent(metrics.accuracy)} correct`
  }
  if (model.task === 'regression' && typeof metrics.r2 === 'number') {
    return `R² ${formatNumber(metrics.r2)}`
  }
  return '—'
}

export function ModelsListPage() {
  const { id = '' } = useParams()
  const dataset = useDataset(id)
  const models = useModels(id)

  if (models.isLoading) return <p className="muted">Loading models…</p>
  if (models.error) return <ErrorBanner error={models.error} />

  const rows = rankModels(models.data ?? [])
  const best = bestModelId(rows)
  const inProgress = rows.filter(
    (m) => m.status === 'queued' || m.status === 'training',
  ).length

  return (
    <div>
      <div className="page-title-row">
        <div>
          <h1>Models — {dataset.data?.name ?? id}</h1>
          <p className="muted">
            Compare runs side by side — for example with and without a column.
            {inProgress > 0 && ` ${inProgress} still training…`}
          </p>
        </div>
        <div className="title-actions">
          <Link className="btn btn-primary" to={`/datasets/${id}/configure`}>
            Train another model
          </Link>
        </div>
      </div>

      {rows.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
          <p className="muted">No models trained on this dataset yet.</p>
          <Link className="btn btn-primary" to={`/datasets/${id}/configure`}>
            Train the first one
          </Link>
        </div>
      ) : (
        <div className="card table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th title="How the run was made: Standard defaults, Thorough search, Fine-tuned knobs, or an auto-compare variant.">Run</th>
                <th>Created</th>
                <th>Predicts</th>
                <th>Type</th>
                <th title="Headline score on rows the model never saw: % correct for category models, R² for number models. Only compare runs predicting the same column.">Quality</th>
                <th>Columns used</th>
                <th>Status</th>
                <th aria-label="Open" />
              </tr>
            </thead>
            <tbody>
              {rows.map((model) => (
                <tr key={model.id}>
                  <td>
                    {variantLabel(model)}
                    {model.id === best && (
                      <span className="chip chip-numeric" style={{ marginLeft: 6 }}>
                        Best
                      </span>
                    )}
                  </td>
                  <td>{new Date(model.created_at).toLocaleString()}</td>
                  <td>
                    <strong>{model.target_column}</strong>
                  </td>
                  <td>{taskLabel(model.task)}</td>
                  <td>{model.status === 'complete' ? keyMetric(model) : '—'}</td>
                  <td>
                    {model.input_spec ? model.input_spec.length : '—'}
                    {model.user_excluded_columns.length > 0 && (
                      <span
                        className="muted small"
                        title={`Left out: ${model.user_excluded_columns.join(', ')}`}
                      >
                        {' '}
                        (−{model.user_excluded_columns.length})
                      </span>
                    )}
                  </td>
                  <td>
                    <span className={`chip ${model.status === 'complete' ? 'chip-numeric' : ''}`}>
                      {model.status}
                    </span>
                  </td>
                  <td>
                    <Link className="btn" to={`/models/${model.id}`}>
                      Open
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
