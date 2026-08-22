import { useInsights } from '../../api/hooks'
import { ApiError } from '../../api/client'
import type { ModelMeta } from '../../api/schemas'
import { ErrorBanner } from '../../components/ErrorBanner'
import { BlueprintCard } from './BlueprintCard'
import { CorrelationHeatmap } from './CorrelationHeatmap'
import { GroupCheckCard } from './GroupCheckCard'
import { BeeswarmLegend, ImpactBeeswarm } from './ImpactBeeswarm'

/**
 * The "Column insights" tab: global explainability computed from the 20% of
 * rows held out during training, so nothing here reflects memorization.
 */
export function InsightsView({ meta }: { meta: ModelMeta }) {
  const insights = useInsights(meta.id, meta.status === 'complete')

  if (insights.isLoading) {
    return <p className="muted">Analyzing how columns relate to the predictions…</p>
  }
  if (insights.error instanceof ApiError && insights.error.code === 'VALIDATION_NOT_FOUND') {
    return (
      <div className="card">
        <h2>Column insights aren’t available for this model</h2>
        <p className="muted">
          This model was trained before insights existed. Train it again and this tab
          will fill in.
        </p>
      </div>
    )
  }
  if (insights.error || !insights.data) {
    return <ErrorBanner error={insights.error} />
  }

  const data = insights.data
  return (
    <div>
      <div className="card">
        <h2>How column values push predictions</h2>
        <p className="muted small">
          Each dot is one of {data.sample_size} rows the model never saw during
          training. Dots to the right pushed that row’s prediction {directionWord(meta)};
          color shows the column’s value for that row.
        </p>
        <BeeswarmLegend />
        <ImpactBeeswarm insights={data} />
      </div>
      <GroupCheckCard meta={meta} />
      <BlueprintCard meta={meta} />
      <div className="card">
        <h2>How columns relate to each other and to the prediction</h2>
        <p className="muted small">
          Darker means more strongly related, measured on {data.association_rows} held-out
          rows. The last row and column show each input’s relationship with{' '}
          <strong>{data.prediction_label.toLowerCase()}</strong>. Numeric pairs also
          show direction: a negative number means one goes up as the other goes down.
        </p>
        <CorrelationHeatmap insights={data} />
      </div>
    </div>
  )
}

function directionWord(meta: ModelMeta): string {
  return meta.task === 'regression' ? 'higher' : 'toward that outcome'
}
