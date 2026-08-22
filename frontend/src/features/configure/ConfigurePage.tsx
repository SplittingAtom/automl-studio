import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'

import {
  useCreateModel,
  useDataset,
  useDatasetAnalysis,
  useDatasetPreview,
  useModel,
  useModels,
} from '../../api/hooks'
import type { Task } from '../../api/schemas'
import { ErrorBanner } from '../../components/ErrorBanner'
import { TabBar } from '../../components/TabBar'
import { WarningBanner } from '../../components/WarningBanner'
import { ExploreView } from '../explore/ExploreView'
import { AnalysisPanel } from './AnalysisPanel'
import { CalculatedColumnCard } from './CalculatedColumnCard'
import { PreviewTable } from './PreviewTable'
import { TargetPicker } from './TargetPicker'
import { experimentPreview } from './experimentPreview'
import {
  EMPTY_TUNING,
  KNOBS,
  toOverrides,
  type MonotoneDirection,
  type TuningState,
} from './tuning'
import { TuningPanel } from './TuningPanel'

type ConfigureView = 'explore' | 'setup'

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

  const [view, setView] = useState<ConfigureView>('explore')
  const [target, setTarget] = useState<string | null>(null)
  const [taskOverride, setTaskOverride] = useState<Task | null>(null)
  const [excluded, setExcluded] = useState<Set<string>>(new Set())
  const [thorough, setThorough] = useState(false)
  const [timeColumn, setTimeColumn] = useState<string>('')
  const [horizon, setHorizon] = useState(0)
  const [tuning, setTuning] = useState<TuningState>(EMPTY_TUNING)

  // "Fine-tune this model": pre-fill everything from the baseline run once.
  const [searchParams] = useSearchParams()
  const baselineId = searchParams.get('baseline')
  const baseline = useModel(baselineId ?? '', baselineId !== null)
  const baselineApplied = useRef(false)
  useEffect(() => {
    const from = baseline.data
    if (!from || from.status !== 'complete' || baselineApplied.current) return
    baselineApplied.current = true
    setView('setup')
    setTarget(from.target_column)
    setTaskOverride(from.task)
    setExcluded(new Set(from.user_excluded_columns))
    setThorough(from.effort === 'thorough')
    setTimeColumn(from.time_column ?? '')
    setHorizon(from.horizon)
    if (from.overrides) {
      const params: TuningState['params'] = {}
      for (const knob of KNOBS) {
        const value = from.overrides[knob.key]
        if (value !== null) params[knob.key] = value
      }
      const monotone = Object.fromEntries(
        Object.entries(from.overrides.monotone_constraints).map(([column, dir]) => [
          column,
          dir as MonotoneDirection,
        ]),
      )
      setTuning({ params, monotone })
    }
  }, [baseline.data])

  if (dataset.isLoading) return <p className="muted">Loading dataset…</p>
  if (dataset.error || !dataset.data) return <ErrorBanner error={dataset.error} />

  const meta = dataset.data
  const targetColumn = meta.columns.find((c) => c.name === target) ?? null
  const detectedTask = targetColumn ? detectTask(targetColumn.kind) : null
  const task = taskOverride ?? detectedTask
  const previousModels = (models.data ?? []).length
  const dateColumns = meta.columns.filter((c) => c.kind === 'datetime')
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

  const baseRequest = () =>
    target && task
      ? {
          dataset_id: meta.id,
          target_column: target,
          task,
          excluded_columns: [...excluded].filter((name) => name !== target),
          time_column: timeColumn || null,
          horizon: timeColumn ? horizon : 0,
        }
      : null

  const handleTrain = () => {
    const base = baseRequest()
    if (!base) return
    createModel.mutate(
      {
        ...base,
        effort: thorough ? 'thorough' : 'standard',
        overrides: toOverrides(tuning),
        baseline_model_id: baselineId ?? undefined,
      },
      { onSuccess: (model) => navigate(`/models/${model.id}`) },
    )
  }

  // Three variants with the same data settings; the leaderboard ranks them.
  const handleAutoCompare = async () => {
    const base = baseRequest()
    if (!base) return
    const variants = [
      { effort: 'standard' as const, label: 'Standard' },
      { effort: 'thorough' as const, label: 'Thorough search' },
      {
        effort: 'standard' as const,
        label: 'Simpler & steadier',
        overrides: { max_depth: 3, reg_lambda: 5 },
      },
    ]
    try {
      for (const variant of variants) {
        await createModel.mutateAsync({ ...base, ...variant })
      }
      navigate(`/datasets/${meta.id}/models`)
    } catch {
      // createModel.error renders in the banner below
    }
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
          {baselineId && baseline.data && (
            <p className="small" style={{ color: 'var(--primary)' }}>
              Fine-tuning from an earlier model — its settings are pre-filled. Training
              creates a new model you can compare against it.
            </p>
          )}
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

      <TabBar<ConfigureView>
        tabs={[
          {
            id: 'explore',
            label: 'Explore data',
            tip: 'Profile every column: distributions, stats, notable patterns, calculated columns, and a raw preview.',
          },
          {
            id: 'setup',
            label: 'Set up model',
            tip: 'Choose what to predict, which columns to use, and how hard to try — then train.',
          },
        ]}
        current={view}
        onSelect={setView}
      />

      {view === 'explore' && (
        <>
          <ExploreView datasetId={meta.id} />
          <CalculatedColumnCard datasetId={meta.id} columns={meta.columns} />
          <div className="card" style={{ marginTop: '1.25rem' }}>
            <h2>Data preview</h2>
            <PreviewTable
              preview={preview.data}
              loading={preview.isLoading}
              error={preview.error}
            />
          </div>
          <div className="explore-continue">
            <button className="btn btn-primary" onClick={() => setView('setup')}>
              Next: choose what to predict →
            </button>
          </div>
        </>
      )}

      {view === 'setup' && (
        <>
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
              {dateColumns.length > 0 && (
                <div className="time-mode-row">
                  <label htmlFor="time-column" className="small" style={{ fontWeight: 600 }}>
                    Time-ordered data?
                  </label>
                  <select
                    id="time-column"
                    value={timeColumn}
                    onChange={(event) => setTimeColumn(event.target.value)}
                  >
                    <option value="">No — rows are independent</option>
                    {dateColumns.map((column) => (
                      <option key={column.name} value={column.name}>
                        Order by {column.name}
                      </option>
                    ))}
                  </select>
                  {timeColumn && (
                    <>
                      <p className="muted small" style={{ margin: '0.25rem 0 0' }}>
                        Tests on the most recent rows and adds recent-history columns —
                        honest evaluation for forecasting-style data.
                      </p>
                      <div className="horizon-row">
                        <label htmlFor="horizon" className="small" style={{ fontWeight: 600 }}>
                          Prediction horizon
                        </label>
                        <input
                          id="horizon"
                          type="number"
                          min={0}
                          max={10000}
                          value={horizon}
                          onChange={(event) =>
                            setHorizon(Math.max(0, Math.floor(Number(event.target.value) || 0)))
                          }
                        />
                        <span className="muted small">rows ahead</span>
                      </div>
                      <p className="muted small" style={{ margin: '0.25rem 0 0' }}>
                        If your target looks N rows into the future (e.g. a 10-day
                        return), set N — a gap that size is left before the test rows so
                        nothing overlaps. Leave 0 for next-row prediction.
                      </p>
                    </>
                  )}
                </div>
              )}
              <label className="effort-toggle">
                <input
                  type="checkbox"
                  checked={thorough}
                  onChange={(event) => setThorough(event.target.checked)}
                />
                Try harder — test a dozen model variations (takes longer)
              </label>
              <TuningPanel
                tuning={tuning}
                onChange={setTuning}
                columns={meta.columns}
                targetColumn={target}
              />
              <div className="experiment-preview">
                <p className="tuning-label" style={{ margin: '0 0 0.25rem' }}>
                  What will happen
                </p>
                <ul className="muted small">
                  {experimentPreview({
                    rowCount: meta.row_count,
                    thorough,
                    timeColumn: timeColumn || null,
                    horizon,
                    tunedSettings:
                      Object.keys(tuning.params).length +
                      Object.keys(tuning.monotone).length,
                  }).map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>
              </div>
              <ErrorBanner error={createModel.error} />
              <button
                className="btn btn-primary"
                style={{ width: '100%', justifyContent: 'center' }}
                onClick={handleTrain}
                disabled={createModel.isPending}
              >
                {createModel.isPending ? 'Starting…' : 'Build model'}
              </button>
              <button
                className="btn"
                style={{ width: '100%', justifyContent: 'center', marginTop: '0.5rem' }}
                onClick={handleAutoCompare}
                disabled={createModel.isPending}
                title="Trains three variants — standard, thorough search, and a simpler steadier one — and ranks them side by side"
              >
                Auto-compare 3 approaches
              </button>
            </>
          )}
        </div>
      </div>
        </>
      )}
    </div>
  )
}
