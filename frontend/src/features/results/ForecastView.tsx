import { useState } from 'react'
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { useForecast, useValidationRows } from '../../api/hooks'
import type { ModelMeta } from '../../api/schemas'
import { ErrorBanner } from '../../components/ErrorBanner'
import { formatNumber } from '../../lib/formatters'
import { appendFuture, buildForecastPoints } from './forecastData'

const CHART_ROWS = 200
const ACTUAL_COLOR = '#94a3b8'
const PREDICTED_COLOR = '#4f46e5'
const BAND_COLOR = '#c7d2fe'

export function ForecastView({ meta }: { meta: ModelMeta }) {
  const [steps, setSteps] = useState(30)
  const validation = useValidationRows(meta.id, meta.status === 'complete', CHART_ROWS)
  const forecast = useForecast(meta.id, steps, meta.status === 'complete')

  if (validation.isLoading) return <p className="muted">Loading the held-out window…</p>
  if (validation.error) return <ErrorBanner error={validation.error} />
  if (!validation.data || !meta.time_column) return null

  const history = buildForecastPoints(
    validation.data.rows,
    meta.time_column,
    meta.target_column,
  )
  const points = appendFuture(history, forecast.data)
  const total = validation.data.total_rows
  const truncated = total > history.length

  return (
    <div className="card">
      <div className="page-title-row">
        <div>
          <h2>Time-series prediction — actual vs predicted</h2>
          <p className="muted small">
            Solid lines: the most recent {total.toLocaleString()} held-back rows,
            predicted blind. Dotted line: the model’s forecast beyond your data, using
            its own predictions as recent history.
          </p>
        </div>
        <div className="horizon-row" style={{ marginTop: 0 }}>
          <label htmlFor="forecast-steps" className="small" style={{ fontWeight: 600 }}>
            Forecast ahead
          </label>
          <input
            id="forecast-steps"
            type="number"
            min={0}
            max={365}
            value={steps}
            onChange={(event) =>
              setSteps(
                Math.max(0, Math.min(365, Math.floor(Number(event.target.value) || 0))),
              )
            }
          />
          <span className="muted small">rows (days here)</span>
        </div>
      </div>
      <ErrorBanner error={forecast.error} />
      <ResponsiveContainer width="100%" height={380}>
        <ComposedChart data={points} margin={{ left: 8, right: 24 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" fontSize={11} minTickGap={48} />
          <YAxis tickFormatter={(value: number) => formatNumber(value)} fontSize={12} width={70} />
          <Tooltip formatter={(value) => formatNumber(value as number)} />
          <Legend />
          <Area
            name="likely range (80%)"
            dataKey="band"
            stroke="none"
            fill={BAND_COLOR}
            fillOpacity={0.5}
            legendType="none"
            tooltipType="none"
          />
          <Line
            name={`actual ${meta.target_column}`}
            dataKey="actual"
            stroke={ACTUAL_COLOR}
            strokeWidth={2}
            dot={false}
          />
          <Line
            name="predicted"
            dataKey="predicted"
            stroke={PREDICTED_COLOR}
            strokeWidth={2}
            dot={false}
          />
          <Line
            name={`forecast (next ${steps})`}
            dataKey="forecast"
            stroke={PREDICTED_COLOR}
            strokeWidth={2}
            strokeDasharray="6 5"
            dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
      <p className="muted small" style={{ marginBottom: 0 }}>
        The forecast assumes the other columns hold their most recent values, and gets
        less certain the further out it goes.
        {truncated &&
          ` Chart shows the first ${history.length} of ${total.toLocaleString()} held-out rows.`}
      </p>
    </div>
  )
}
