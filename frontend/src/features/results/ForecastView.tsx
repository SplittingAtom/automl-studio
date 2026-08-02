import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { useValidationRows } from '../../api/hooks'
import type { ModelMeta } from '../../api/schemas'
import { ErrorBanner } from '../../components/ErrorBanner'
import { formatNumber } from '../../lib/formatters'
import { buildForecastPoints } from './forecastData'

const CHART_ROWS = 200
const ACTUAL_COLOR = '#94a3b8'
const PREDICTED_COLOR = '#4f46e5'

export function ForecastView({ meta }: { meta: ModelMeta }) {
  const validation = useValidationRows(meta.id, meta.status === 'complete', CHART_ROWS)

  if (validation.isLoading) return <p className="muted">Loading the held-out window…</p>
  if (validation.error) return <ErrorBanner error={validation.error} />
  if (!validation.data || !meta.time_column) return null

  const points = buildForecastPoints(
    validation.data.rows,
    meta.time_column,
    meta.target_column,
  )
  const total = validation.data.total_rows
  const truncated = total > points.length

  return (
    <div className="card">
      <h2>Time-series prediction — actual vs predicted</h2>
      <p className="muted small">
        The most recent {total.toLocaleString()} rows were held back from training; the
        model predicted them blind, in order. The closer the lines track, the better the
        forecast.
      </p>
      <ResponsiveContainer width="100%" height={360}>
        <LineChart data={points} margin={{ left: 8, right: 24 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" fontSize={11} minTickGap={48} />
          <YAxis tickFormatter={(value: number) => formatNumber(value)} fontSize={12} width={70} />
          <Tooltip formatter={(value) => formatNumber(value as number)} />
          <Legend />
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
        </LineChart>
      </ResponsiveContainer>
      {truncated && (
        <p className="muted small" style={{ marginBottom: 0 }}>
          Showing the first {points.length} of {total.toLocaleString()} held-out rows —
          download the full set from the validation table on the Overview tab.
        </p>
      )}
    </div>
  )
}
