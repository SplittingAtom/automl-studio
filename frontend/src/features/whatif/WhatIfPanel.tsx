import { useMemo } from 'react'

import { usePrediction } from '../../api/hooks'
import type { ModelMeta } from '../../api/schemas'
import { ErrorBanner } from '../../components/ErrorBanner'
import { useDebouncedValue } from '../../lib/useDebouncedValue'
import { CategorySelect } from './CategorySelect'
import { NumericSlider } from './NumericSlider'
import { PredictionDisplay } from './PredictionDisplay'
import { useWhatIfState } from './useWhatIfState'

const DEBOUNCE_MS = 250

export function WhatIfPanel({ model }: { model: ModelMeta }) {
  const spec = useMemo(() => model.input_spec ?? [], [model.input_spec])
  const { values, setValue, reset } = useWhatIfState(spec)
  const debouncedValues = useDebouncedValue(values, DEBOUNCE_MS)
  const prediction = usePrediction(model.id, debouncedValues, spec.length > 0)

  const updating =
    prediction.isFetching ||
    JSON.stringify(values) !== JSON.stringify(debouncedValues)

  return (
    <div className="card">
      <h2>Explore what-if scenarios</h2>
      <p className="muted small">
        Adjust the inputs and watch the prediction update in real time.
      </p>

      <PredictionDisplay
        prediction={prediction.data}
        targetColumn={model.target_column}
        updating={updating}
      />
      <ErrorBanner error={prediction.error} />

      {spec.map((item) =>
        item.kind === 'numeric' ? (
          <NumericSlider
            key={item.name}
            item={item}
            value={Number(values[item.name] ?? 0)}
            onChange={(value) => setValue(item.name, value)}
          />
        ) : (
          <CategorySelect
            key={item.name}
            item={item}
            value={String(values[item.name] ?? '')}
            onChange={(value) => setValue(item.name, value)}
          />
        ),
      )}

      <button className="btn" onClick={reset}>
        Reset to typical values
      </button>
    </div>
  )
}
