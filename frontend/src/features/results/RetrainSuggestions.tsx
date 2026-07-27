import { useNavigate } from 'react-router-dom'

import { useCreateModel } from '../../api/hooks'
import type { ModelMeta } from '../../api/schemas'
import { ErrorBanner } from '../../components/ErrorBanner'

export function RetrainSuggestions({ meta }: { meta: ModelMeta }) {
  const navigate = useNavigate()
  const createModel = useCreateModel()

  const alreadyExcluded = new Set(meta.user_excluded_columns)
  const lowImportance = meta.suggested_exclusions.filter((c) => !alreadyExcluded.has(c))
  const leak = meta.leak_suspect

  if (!leak && lowImportance.length === 0) return null

  const retrainWithout = (columns: string[]) => {
    createModel.mutate(
      {
        dataset_id: meta.dataset_id,
        target_column: meta.target_column,
        task: meta.task,
        excluded_columns: [...new Set([...meta.user_excluded_columns, ...columns])],
      },
      { onSuccess: (model) => navigate(`/models/${model.id}`) },
    )
  }

  return (
    <div className="card suggestions-card">
      <h2>Suggestions</h2>
      <ErrorBanner error={createModel.error} />
      {leak && (
        <div className="suggestion-row">
          <span>
            <strong>“{leak}”</strong> may already contain the answer — the model leans on
            it almost exclusively. Removing it will show what the rest of your data can do.
          </span>
          <button
            className="btn"
            disabled={createModel.isPending}
            onClick={() => retrainWithout([leak])}
          >
            Retrain without it
          </button>
        </div>
      )}
      {lowImportance.length > 0 && (
        <div className="suggestion-row">
          <span>
            {lowImportance.length > 1 ? 'These columns' : 'This column'} barely influenced
            the model: <strong>{lowImportance.join(', ')}</strong>. Removing{' '}
            {lowImportance.length > 1 ? 'them' : 'it'} simplifies the what-if panel — use
            “Compare models” afterwards to confirm nothing was lost.
          </span>
          <button
            className="btn"
            disabled={createModel.isPending}
            onClick={() => retrainWithout(lowImportance)}
          >
            Retrain without {lowImportance.length > 1 ? 'them' : 'it'}
          </button>
        </div>
      )}
      {createModel.isPending && <p className="muted small">Starting a new training run…</p>}
    </div>
  )
}
