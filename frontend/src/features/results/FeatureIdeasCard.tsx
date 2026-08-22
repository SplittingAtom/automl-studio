import { useNavigate } from 'react-router-dom'

import { useCreateCalculatedColumn, useFeatureIdeas } from '../../api/hooks'
import type { ModelMeta } from '../../api/schemas'
import { ErrorBanner } from '../../components/ErrorBanner'

/**
 * Bounded automatic feature engineering: simple formulas built from the
 * model's strongest columns that a probe fit found genuinely useful. One
 * click adds the column (as a new derived dataset) and opens setup.
 */
export function FeatureIdeasCard({ meta }: { meta: ModelMeta }) {
  const navigate = useNavigate()
  const ideas = useFeatureIdeas(meta.id, meta.status === 'complete')
  const addColumn = useCreateCalculatedColumn(meta.dataset_id)

  if (ideas.isLoading || ideas.error) return null // quietly absent, never blocking
  const found = ideas.data?.ideas ?? []
  if (found.length === 0) return null

  const handleAdd = (name: string, formula: string) => {
    addColumn.mutate(
      { name, formula },
      { onSuccess: (dataset) => navigate(`/datasets/${dataset.id}/configure`) },
    )
  }

  return (
    <div className="card">
      <h2>Column ideas</h2>
      <p className="muted small">
        Simple combinations of your strongest columns that a quick test found
        useful. Adding one creates a copy of the dataset so you can retrain and
        compare.
      </p>
      <ErrorBanner error={addColumn.error} />
      {found.map((idea) => (
        <div key={idea.name} className="idea-row">
          <div>
            <code className="idea-formula">{idea.formula}</code>
            <span className="muted small">
              {' '}
              — carried {Math.round(idea.share * 100)}% of a test model's decisions
            </span>
          </div>
          <button
            className="btn"
            onClick={() => handleAdd(idea.name, idea.formula)}
            disabled={addColumn.isPending}
          >
            {addColumn.isPending ? 'Adding…' : 'Add column'}
          </button>
        </div>
      ))}
    </div>
  )
}
