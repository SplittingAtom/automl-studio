import type { PredictResponse } from '../../api/schemas'
import { formatNumber, formatPercent } from '../../lib/formatters'

export function PredictionDisplay({
  prediction,
  targetColumn,
  updating,
}: {
  prediction: PredictResponse | undefined
  targetColumn: string
  updating: boolean
}) {
  if (!prediction) {
    return (
      <div className="prediction-display">
        <span className="muted">Calculating…</span>
      </div>
    )
  }

  const value =
    typeof prediction.prediction === 'number'
      ? formatNumber(prediction.prediction)
      : prediction.prediction

  return (
    <div className={`prediction-display${updating ? ' updating' : ''}`}>
      <div className="muted small">Predicted {targetColumn}</div>
      <div className="value">{value}</div>
      {prediction.probabilities && (
        <div>
          {prediction.probabilities.map((entry) => (
            <div key={entry.label} className="proba-row">
              <span className="proba-label" title={entry.label}>
                {entry.label}
              </span>
              <div className="bar-track">
                <div
                  className="bar-fill"
                  style={{ width: `${Math.round(entry.probability * 100)}%` }}
                />
              </div>
              <span className="proba-value">{formatPercent(entry.probability)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
