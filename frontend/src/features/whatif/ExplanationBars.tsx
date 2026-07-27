import type { Explanation } from '../../api/schemas'

const MAX_ITEMS = 6

export function ExplanationBars({
  explanation,
  isClassification,
}: {
  explanation: Explanation
  isClassification: boolean
}) {
  const items = explanation.items.slice(0, MAX_ITEMS)
  const largest = Math.max(...items.map((i) => Math.abs(i.contribution)), 1e-9)
  const caption = isClassification
    ? `What pushed this prediction toward or away from “${explanation.toward_label}”`
    : 'What pushed this prediction up or down'

  return (
    <div className="explanation">
      <div className="muted small" style={{ marginBottom: '0.4rem' }}>
        {caption}
      </div>
      {items.map((item) => {
        const width = Math.max(3, Math.round((Math.abs(item.contribution) / largest) * 100))
        const positive = item.contribution >= 0
        return (
          <div key={item.feature} className="explanation-row">
            <span className="explanation-label" title={item.feature}>
              {item.feature}
            </span>
            <div className="explanation-track">
              <div
                className={`explanation-fill ${positive ? 'positive' : 'negative'}`}
                style={{ width: `${width}%` }}
              />
            </div>
            <span className={`explanation-sign ${positive ? 'positive' : 'negative'}`}>
              {positive ? '▲' : '▼'}
            </span>
          </div>
        )
      })}
    </div>
  )
}
