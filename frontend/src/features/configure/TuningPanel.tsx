import type { ColumnProfile } from '../../api/schemas'
import {
  KNOBS,
  type KnobKey,
  type MonotoneDirection,
  type TuningState,
} from './tuning'

export function TuningPanel({
  tuning,
  onChange,
  columns,
  targetColumn,
}: {
  tuning: TuningState
  onChange: (next: TuningState) => void
  columns: ColumnProfile[]
  targetColumn: string | null
}) {
  const numericColumns = columns.filter(
    (c) => c.kind === 'numeric' && c.name !== targetColumn,
  )
  const activeCount =
    Object.keys(tuning.params).length + Object.keys(tuning.monotone).length

  const setParam = (key: KnobKey, value: number | undefined) => {
    const params = { ...tuning.params }
    if (value === undefined) delete params[key]
    else params[key] = value
    onChange({ ...tuning, params })
  }

  const setMonotone = (column: string, direction: MonotoneDirection | undefined) => {
    const monotone = { ...tuning.monotone }
    if (direction === undefined) delete monotone[column]
    else monotone[column] = direction
    onChange({ ...tuning, monotone })
  }

  return (
    <details className="tuning-panel">
      <summary>
        Advanced tuning{' '}
        <span className="muted small">
          {activeCount > 0 ? `· ${activeCount} setting${activeCount === 1 ? '' : 's'} changed` : '· optional'}
        </span>
      </summary>
      <p className="muted small">
        Every knob has a safe range, and each run is a new model — your baseline stays
        untouched, so you can always compare and go back.
      </p>
      {KNOBS.map((knob) => {
        const active = tuning.params[knob.key] !== undefined
        const value = tuning.params[knob.key] ?? knob.defaultValue
        return (
          <div key={knob.key} className="tuning-row">
            <label className="tuning-row-head">
              <input
                type="checkbox"
                checked={active}
                onChange={(event) =>
                  setParam(knob.key, event.target.checked ? value : undefined)
                }
              />
              <span className="tuning-label">{knob.label}</span>
              <code className="tuning-code">{knob.key}</code>
              <span className="tuning-value">
                {active ? value : `${knob.defaultValue} (default)`}
              </span>
            </label>
            {active && (
              <>
                <input
                  type="range"
                  min={knob.min}
                  max={knob.max}
                  step={knob.step}
                  value={value}
                  onChange={(event) => setParam(knob.key, Number(event.target.value))}
                />
                <p className="muted small tuning-hint">{knob.hint}</p>
              </>
            )}
          </div>
        )
      })}
      {numericColumns.length > 0 && (
        <div className="tuning-monotone">
          <p className="tuning-label" style={{ margin: '0.75rem 0 0.25rem' }}>
            Direction rules
          </p>
          <p className="muted small">
            Force the prediction to only ever rise (or fall) as a column grows — useful
            when you know the real-world relationship, and it makes the model easier to
            trust.
          </p>
          {numericColumns.map((column) => (
            <div key={column.name} className="tuning-monotone-row">
              <span className="tuning-label" title={column.name}>
                {column.name}
              </span>
              <select
                value={tuning.monotone[column.name] ?? 0}
                onChange={(event) => {
                  const raw = Number(event.target.value)
                  setMonotone(
                    column.name,
                    raw === 0 ? undefined : (raw as MonotoneDirection),
                  )
                }}
              >
                <option value={0}>No rule</option>
                <option value={1}>Prediction only rises with it</option>
                <option value={-1}>Prediction only falls with it</option>
              </select>
            </div>
          ))}
        </div>
      )}
    </details>
  )
}
