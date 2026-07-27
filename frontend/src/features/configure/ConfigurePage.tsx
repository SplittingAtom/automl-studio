import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'

import {
  useCreateModel,
  useDataset,
  useDatasetAnalysis,
  useDatasetPreview,
  useModels,
} from '../../api/hooks'
import type { Task } from '../../api/schemas'
import { ErrorBanner } from '../../components/ErrorBanner'
import { WarningBanner } from '../../components/WarningBanner'
import { AnalysisPanel } from './AnalysisPanel'
import { CalculatedColumnCard } from './CalculatedColumnCard'
import { PreviewTable } from './PreviewTable'
import { TargetPicker } from './TargetPicker'

/** Mirrors the backend heuristic so the UI can show the detected task instantly. */
function detectTask(kind: string): Task {
  return kind === 'numeric' ? 'regression' : 'classification'
}

export function ConfigurePage() {
  const { id = '' } = useParams()
  const navigate = useNavigate()
  const dataset = useDataset(id)
  const preview = useDatasetPreview(id)
  const createModel = useCreateModel()

  const models = useModels(id)
  const analysis = useDatasetAnalysis(id)

  const [target, setTarget] = useState<string | null>(null)
  const [taskOverride, setTaskOverride] = useState<Task | null>(null)
  const [excluded, setExcluded] = useState<Set<string>>(new Set())
  const [thorough, setThorough] = useState(false)

  if (dataset.isLoading) return <p className="muted">Loading dataset…</p>
  if (dataset.error || !dataset.data) return <ErrorBanner error={dataset.error} />

  const meta = dataset.data
  const targetColumn = meta.columns.find((c) => c.name === target) ?? null
  const detectedTask = targetColumn ? detectTask(targetColumn.kind) : null
  const task = taskOverride ?? detectedTask
  const previousModels = (models.data ?? []).length
  const recommended = new Set(
    (analysis.data?.candidates ?? []).filter((c) => c.recommended).map((c) => c.column),
  )
  const targetCandidate =
    (analysis.data?.candidates ?? []).find((c) => c.column === target) ?? null

  const toggleUse = (name: string) => {
    setExcluded((current) => {
      const next = new Set(current)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }

  const handleTrain = () => {
    if (!target || !task) return
    createModel.mutate(
      {
        dataset_id: meta.id,
        target_column: target,
        task,
        excluded_columns: [...excluded].filter((name) => name !== target),
        effort: thorough ? 'thorough' : 'standard',
      },
      { onSuccess: (model) => navigate(`/models/${model.id}`) },
    )
  }

  return (
    <div>
      <div className="page-title-row">
        <div>
          <h1>{meta.name}</h1>
          <p className="muted">
            {meta.row_count.toLocaleString()} rows · {meta.column_count} columns. Choose
            the column you want the model to predict.
          </p>
        </div>
        {previousModels > 0 && (
          <div className="title-actions">
            <Link className="btn" to={`/datasets/${meta.id}/models`}>
              Previous models ({previousModels})
            </Link>
          </div>
        )}
      </div>
      <WarningBanner warnings={meta.warnings} />

      <AnalysisPanel
        analysis={analysis.data}
        loading={analysis.isLoading}
        error={analysis.error}
        selectedTarget={target}
        onPickTarget={(candidate) => {
          setTarget(candidate.column)
          setTaskOverride(null)
        }}
      />

      <div className="configure-grid" style={{ marginTop: '1.25rem' }}>
        <div className="card">
          <h2>Pick a column to predict</h2>
          <TargetPicker
            columns={meta.columns}
            selected={target}
            excluded={excluded}
            recommended={recommended}
            onSelect={(name) => {
              setTarget(name)
              setTaskOverride(null)
            }}
            onToggleUse={toggleUse}
          />
        </div>

        <div className="card">
          <h2>Model setup</h2>
          {!targetColumn && (
            <p className="muted small">Select a column on the left to continue.</p>
          )}
          {targetColumn && detectedTask && (
            <>
              <p className="small">
                Predicting <strong>{targetColumn.name}</strong>
              </p>
              <div className="radio-row" role="radiogroup" aria-label="Prediction type">
                {(['classification', 'regression'] as const).map((option) => (
                  <label key={option} className={task === option ? 'checked' : ''}>
                    <input
                      type="radio"
                      name="task"
                      checked={task === option}
                      onChange={() => setTaskOverride(option)}
                    />
                    {option === 'classification' ? 'Predict a category' : 'Predict a number'}
                  </label>
                ))}
              </div>
              {taskOverride === null && (
                <p className="muted small">
                  Detected automatically — override it if that looks wrong.
                </p>
              )}
              {targetCandidate && targetCandidate.top_predictors.length > 0 && (
                <p className="muted small">
                  Strongest predictors:{' '}
                  {targetCandidate.top_predictors.slice(0, 3).map((p) => p.name).join(', ')}
                </p>
              )}
              {excluded.size > 0 && (
                <p className="muted small">
                  Leaving out: {[...excluded].filter((n) => n !== target).join(', ')}
                </p>
              )}
              <label className="effort-toggle">
                <input
                  type="checkbox"
                  checked={thorough}
                  onChange={(event) => setThorough(event.target.checked)}
                />
                Try harder — test a dozen model variations (takes longer)
              </label>
              <ErrorBanner error={createModel.error} />
              <button
                className="btn btn-primary"
                style={{ width: '100%', justifyContent: 'center' }}
                onClick={handleTrain}
                disabled={createModel.isPending}
              >
                {createModel.isPending ? 'Starting…' : 'Build model'}
              </button>
            </>
          )}
        </div>
      </div>

      <CalculatedColumnCard datasetId={meta.id} columns={meta.columns} />

      <div className="card" style={{ marginTop: '1.25rem' }}>
        <h2>Data preview</h2>
        <PreviewTable preview={preview.data} loading={preview.isLoading} error={preview.error} />
      </div>
    </div>
  )
}
