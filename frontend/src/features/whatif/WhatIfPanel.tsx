import type { ModelMeta, PredictResponse, WhatIfValues } from '../../api/schemas'
import { ErrorBanner } from '../../components/ErrorBanner'
import { InfoTip } from '../../components/InfoTip'
import { CategorySelect } from './CategorySelect'
import { ExplanationBars } from './ExplanationBars'
import { NumericSlider } from './NumericSlider'
import { PredictionDisplay } from './PredictionDisplay'
import { outOfRangeInputs } from './rangeCheck'

export function WhatIfPanel({
  model,
  values,
  setValue,
  reset,
  prediction,
  error,
  updating,
}: {
  model: ModelMeta
  values: WhatIfValues
  setValue: (name: string, value: number | string) => void
  reset: () => void
  prediction: PredictResponse | undefined
  error: unknown
  updating: boolean
}) {
  const spec = model.input_spec ?? []
  const outOfRange = outOfRangeInputs(spec, values)

  return (
    <div className="card">
      <h2>
        Explore what-if scenarios
        <InfoTip tip="Ask the model hypothetical questions: set any combination of inputs and it predicts live. The bars underneath show which of your values pushed the prediction up or down." />
      </h2>
      <p className="muted small">
        Adjust the inputs and watch the prediction update in real time.
      </p>

      <PredictionDisplay
        prediction={prediction}
        targetColumn={model.target_column}
        updating={updating}
      />
      {outOfRange.length > 0 && (
        <div className="range-note" role="note">
          Exploring beyond your data: <strong>{outOfRange.join(', ')}</strong>{' '}
          {outOfRange.length > 1 ? 'are' : 'is'} outside the range the model was trained
          on. The model can't extrapolate out here, so treat this prediction with caution.
        </div>
      )}
      {prediction?.explanation && (
        <ExplanationBars
          explanation={prediction.explanation}
          isClassification={model.task === 'classification'}
        />
      )}
      <ErrorBanner error={error} />

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
