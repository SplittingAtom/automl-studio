import { useMemo } from 'react'

import { usePrediction } from '../../api/hooks'
import { InfoTip } from '../../components/InfoTip'
import type { ModelMeta } from '../../api/schemas'
import { useDebouncedValue } from '../../lib/useDebouncedValue'
import { SensitivityCard } from '../sensitivity/SensitivityCard'
import { WhatIfPanel } from '../whatif/WhatIfPanel'
import { useWhatIfState } from '../whatif/useWhatIfState'
import { CalibrationCard } from './CalibrationCard'
import { FeatureIdeasCard } from './FeatureIdeasCard'
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
        <CalibrationCard meta={meta} />
        <div className="card">
          <h2>
            What drives the predictions?
            <InfoTip tip="Feature importance: how much each column improved the model's decisions during training, scaled so all columns sum to 100%. A dominant single column can be a red flag — it may already contain the answer." />
          </h2>
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
        <FeatureIdeasCard meta={meta} />
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
