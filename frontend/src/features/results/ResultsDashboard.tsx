import { useMemo } from 'react'

import { usePrediction } from '../../api/hooks'
import type { ModelMeta } from '../../api/schemas'
import { useDebouncedValue } from '../../lib/useDebouncedValue'
import { SensitivityCard } from '../sensitivity/SensitivityCard'
import { WhatIfPanel } from '../whatif/WhatIfPanel'
import { useWhatIfState } from '../whatif/useWhatIfState'
import { ImportanceChart } from './ImportanceChart'
import { MetricsCards } from './MetricsCards'
import { ThresholdCard } from './ThresholdCard'
import { ValidationCard } from './ValidationCard'

const DEBOUNCE_MS = 250

export function ResultsDashboard({ meta }: { meta: ModelMeta }) {
  const spec = useMemo(() => meta.input_spec ?? [], [meta.input_spec])
  const { values, setValue, reset } = useWhatIfState(spec)
  const debouncedValues = useDebouncedValue(values, DEBOUNCE_MS)
  const prediction = usePrediction(meta.id, debouncedValues, spec.length > 0)
  const updating =
    prediction.isFetching || JSON.stringify(values) !== JSON.stringify(debouncedValues)

  return (
    <div>
      <div className="results-grid">
      <div>
        <MetricsCards meta={meta} />
        <ThresholdCard meta={meta} />
        <div className="card">
          <h2>What drives the predictions?</h2>
          <p className="muted small">
            The share of the model’s decisions each column is responsible for.
          </p>
          <ImportanceChart importance={meta.importance ?? []} />
          {meta.excluded_columns.length > 0 && (
            <p className="muted small" style={{ marginBottom: 0 }}>
              Not used: {meta.excluded_columns.map((c) => c.name).join(', ')}
            </p>
          )}
        </div>
        <SensitivityCard meta={meta} inputs={debouncedValues} />
      </div>
      <WhatIfPanel
        model={meta}
        values={values}
        setValue={setValue}
        reset={reset}
        prediction={prediction.data}
        error={prediction.error}
        updating={updating}
      />
      </div>
      <ValidationCard meta={meta} />
    </div>
  )
}
