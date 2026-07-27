import type { DatasetAnalysis, TargetCandidate } from '../../api/schemas'
import { ErrorBanner } from '../../components/ErrorBanner'
import { taskLabel } from '../../lib/formatters'

const RATING_LABELS: Record<DatasetAnalysis['rating'], string> = {
  great: 'Great fit',
  good: 'Good fit',
  fair: 'Workable',
  poor: 'Poor fit',
}

const TONE_ICONS: Record<'good' | 'warn' | 'bad', string> = {
  good: '✓',
  warn: '⚠',
  bad: '✕',
}

function scoreLabel(score: number): string {
  if (score >= 75) return 'Excellent'
  if (score >= 55) return 'Good'
  if (score >= 40) return 'Fair'
  return 'Weak'
}

export function AnalysisPanel({
  analysis,
  loading,
  error,
  selectedTarget,
  onPickTarget,
}: {
  analysis: DatasetAnalysis | undefined
  loading: boolean
  error: unknown
  selectedTarget: string | null
  onPickTarget: (candidate: TargetCandidate) => void
}) {
  if (loading) {
    return (
      <div className="card analysis-card">
        <div className="analysis-loading">
          <div className="spinner" />
          <div>
            <h2>Analyzing your dataset…</h2>
            <p className="muted small" style={{ marginBottom: 0 }}>
              Checking every column for how predictable it is from the others.
            </p>
          </div>
        </div>
      </div>
    )
  }
  if (error) return <ErrorBanner error={error} />
  if (!analysis) return null

  // Real recommendations first; derived look-alikes sink to the bottom
  const shown = [...analysis.candidates]
    .sort((a, b) =>
      Number(b.recommended) - Number(a.recommended) ||
      Number(a.derived_like) - Number(b.derived_like) ||
      b.score - a.score,
    )
    .slice(0, 6)

  return (
    <div className="card analysis-card">
      <div className="analysis-header">
        <h2 style={{ marginBottom: 0 }}>Dataset analysis</h2>
        <span className={`chip rating-${analysis.rating}`}>
          {RATING_LABELS[analysis.rating]}
        </span>
      </div>
      <p style={{ marginTop: '0.4rem' }}>{analysis.summary}</p>
      <ul className="analysis-points">
        {analysis.points.map((point) => (
          <li key={point.message} className={`tone-${point.tone}`}>
            <span className="tone-icon">{TONE_ICONS[point.tone]}</span> {point.message}
          </li>
        ))}
      </ul>

      {shown.length > 0 && (
        <>
          <h3 style={{ marginTop: '1rem' }}>What could you predict?</h3>
          <div className="candidate-list">
            {shown.map((candidate) => (
              <button
                key={candidate.column}
                className={`candidate-row${selectedTarget === candidate.column ? ' selected' : ''}`}
                onClick={() => onPickTarget(candidate)}
              >
                <div className="candidate-main">
                  <strong>{candidate.column}</strong>
                  {candidate.recommended && <span className="chip chip-numeric">Recommended</span>}
                  {candidate.derived_like && (
                    <span className="chip score-fair">Possibly derived</span>
                  )}
                  <span className="chip">{taskLabel(candidate.task)}</span>
                  <span className={`chip score-${scoreLabel(candidate.score).toLowerCase()}`}>
                    {scoreLabel(candidate.score)} · {candidate.score}
                  </span>
                </div>
                <div className="muted small">{candidate.reasons[0]}</div>
                {candidate.top_predictors.length > 0 && (
                  <div className="muted small">
                    Best predictors: {candidate.top_predictors.slice(0, 3).map((p) => p.name).join(', ')}
                  </div>
                )}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
