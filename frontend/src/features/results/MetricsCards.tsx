import type { ModelMeta } from '../../api/schemas'
import { formatNumber, formatPercent } from '../../lib/formatters'

interface MetricCard {
  value: string
  label: string
  tip: string
}

function asNumber(value: unknown): number | null {
  return typeof value === 'number' ? value : null
}

function classificationCards(metrics: Record<string, unknown>): MetricCard[] {
  const accuracy = asNumber(metrics.accuracy)
  const f1 = asNumber(metrics.f1_weighted)
  const auc = asNumber(metrics.roc_auc)
  const cards: MetricCard[] = []
  if (accuracy !== null) {
    cards.push({
      value: formatPercent(accuracy),
      label: 'of test predictions were correct',
      tip: 'Accuracy: the share of held-out test rows the model classified correctly. Compare it with the "always guess the most common value" number below — that gap is the model\'s real contribution.',
    })
  }
  if (f1 !== null) {
    cards.push({
      value: formatPercent(f1),
      label: 'balanced score across categories (F1)',
      tip: 'F1 (weighted): balances catching each category against false alarms, so a model can\'t look good by only ever predicting the common category. Most useful when one category is rare.',
    })
  }
  if (auc !== null) {
    cards.push({
      value: formatPercent(auc),
      label: 'ability to tell the categories apart (AUC)',
      tip: 'ROC AUC: how well the model ranks cases — 50% is a coin flip, 100% is perfect separation. Unlike accuracy, it doesn\'t depend on the decision cut-off.',
    })
  }
  return cards
}

function regressionCards(metrics: Record<string, unknown>): MetricCard[] {
  const r2 = asNumber(metrics.r2)
  const mae = asNumber(metrics.mae)
  const mean = asNumber(metrics.target_mean)
  const cards: MetricCard[] = []
  if (r2 !== null) {
    cards.push({
      value: formatPercent(Math.max(0, r2)),
      label: 'of the variation is explained by the model (R²)',
      tip: 'R²: how much of the target\'s ups and downs the model accounts for. 100% is perfect; 0% means no better than always predicting the average.',
    })
  }
  if (mae !== null) {
    cards.push({
      value: formatNumber(mae),
      label:
        mean !== null
          ? `typical prediction error (average value is ${formatNumber(mean)})`
          : 'typical prediction error',
      tip: 'MAE (mean absolute error): on average, how far predictions land from the true value, in the same units as the target. Judge it against the average value shown.',
    })
  }
  return cards
}

function baselineContext(meta: ModelMeta): string | null {
  const metrics = meta.metrics ?? {}
  if (meta.task === 'classification') {
    const naive = asNumber(metrics.baseline_accuracy)
    if (naive === null) return null
    const parts = [`always guessing the most common value gets ${formatPercent(naive)}`]
    const linear = asNumber(metrics.linear_accuracy)
    if (linear !== null) parts.push(`a basic statistical model gets ${formatPercent(linear)}`)
    return `For context: ${parts.join(' · ')}.`
  }
  const naiveMae = asNumber(metrics.baseline_mae)
  if (naiveMae === null) return null
  const parts = [`always predicting the average is off by ${formatNumber(naiveMae)}`]
  const linearMae = asNumber(metrics.linear_mae)
  if (linearMae !== null) parts.push(`a basic statistical model is off by ${formatNumber(linearMae)}`)
  return `For context: ${parts.join(' · ')}.`
}

function consistencyLine(meta: ModelMeta): string | null {
  const metrics = meta.metrics ?? {}
  const mean = asNumber(metrics.cv_mean)
  const std = asNumber(metrics.cv_std)
  const folds = asNumber(metrics.cv_folds)
  if (mean === null || std === null) return null
  const spread = `${formatPercent(Math.max(0, mean - std))}–${formatPercent(Math.min(1, mean + std))}`
  if (meta.task === 'classification') {
    return `Consistency check (${folds}-fold): expect roughly ${spread} correct on new data.`
  }
  return `Consistency check (${folds}-fold): expect the model to explain roughly ${spread} of the variation on new data.`
}

export function MetricsCards({ meta }: { meta: ModelMeta }) {
  const metrics = meta.metrics ?? {}
  const cards =
    meta.task === 'classification' ? classificationCards(metrics) : regressionCards(metrics)
  const testRows = asNumber(metrics.test_rows)
  const context = baselineContext(meta)
  const consistency = consistencyLine(meta)

  return (
    <div>
      <div className="metric-cards">
        {cards.map((card) => (
          <div key={card.label} className="card metric-card" title={card.tip}>
            <div className="value">{card.value}</div>
            <div className="label">{card.label}</div>
          </div>
        ))}
        {testRows !== null && (
          <div
            className="card metric-card"
            title="These rows were split off before training and never shown to the model — scoring on them estimates how it will do on genuinely new data."
          >
            <div className="value">{testRows.toLocaleString()}</div>
            <div className="label">rows held back to test the model fairly</div>
          </div>
        )}
      </div>
      {context && <p className="muted small context-line">{context}</p>}
      {consistency && <p className="muted small context-line">{consistency}</p>}
    </div>
  )
}
