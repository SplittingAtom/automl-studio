import type { ModelMeta } from '../../api/schemas'
import { formatNumber, formatPercent } from '../../lib/formatters'

interface MetricCard {
  value: string
  label: string
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
    cards.push({ value: formatPercent(accuracy), label: 'of test predictions were correct' })
  }
  if (f1 !== null) {
    cards.push({ value: formatPercent(f1), label: 'balanced score across categories (F1)' })
  }
  if (auc !== null) {
    cards.push({ value: formatPercent(auc), label: 'ability to tell the categories apart (AUC)' })
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
    })
  }
  if (mae !== null) {
    cards.push({
      value: formatNumber(mae),
      label:
        mean !== null
          ? `typical prediction error (average value is ${formatNumber(mean)})`
          : 'typical prediction error',
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

export function MetricsCards({ meta }: { meta: ModelMeta }) {
  const metrics = meta.metrics ?? {}
  const cards =
    meta.task === 'classification' ? classificationCards(metrics) : regressionCards(metrics)
  const testRows = asNumber(metrics.test_rows)
  const context = baselineContext(meta)

  return (
    <div>
      <div className="metric-cards">
        {cards.map((card) => (
          <div key={card.label} className="card metric-card">
            <div className="value">{card.value}</div>
            <div className="label">{card.label}</div>
          </div>
        ))}
        {testRows !== null && (
          <div className="card metric-card">
            <div className="value">{testRows.toLocaleString()}</div>
            <div className="label">rows held back to test the model fairly</div>
          </div>
        )}
      </div>
      {context && <p className="muted small context-line">{context}</p>}
    </div>
  )
}
