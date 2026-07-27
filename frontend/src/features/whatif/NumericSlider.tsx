import type { InputSpecItem } from '../../api/schemas'
import { formatNumber } from '../../lib/formatters'

const RANGE_PADDING = 0.05
const SLIDER_STEPS = 200

export function NumericSlider({
  item,
  value,
  onChange,
}: {
  item: InputSpecItem
  value: number
  onChange: (value: number) => void
}) {
  const baseMin = item.min_value ?? 0
  const baseMax = item.max_value ?? 100
  const pad = (baseMax - baseMin) * RANGE_PADDING
  // Don't pad past zero for non-negative columns (age, price, count…)
  const min = baseMin >= 0 ? Math.max(0, baseMin - pad) : baseMin - pad
  const max = baseMax + pad
  const step = (max - min) / SLIDER_STEPS || 1

  const outOfRange = value < baseMin || value > baseMax

  return (
    <div className="whatif-control">
      <div className="control-header">
        <label htmlFor={`slider-${item.name}`}>{item.name}</label>
        <input
          type="number"
          className={outOfRange ? 'out-of-range' : undefined}
          title={outOfRange ? 'Outside the range the model was trained on' : undefined}
          aria-label={`${item.name} exact value`}
          value={Number.isFinite(value) ? Number(value.toFixed(4)) : ''}
          onChange={(event) => {
            const parsed = Number(event.target.value)
            if (Number.isFinite(parsed)) onChange(parsed)
          }}
        />
      </div>
      <input
        id={`slider-${item.name}`}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <div className="control-header muted small">
        <span>{formatNumber(min)}</span>
        <span>{formatNumber(max)}</span>
      </div>
    </div>
  )
}
