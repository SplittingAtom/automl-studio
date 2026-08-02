import type { ForecastResponse } from '../../api/schemas'

export interface ForecastPoint {
  date: string
  actual?: number
  predicted?: number
  forecast?: number
  band?: [number, number]
}

function asFiniteNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === '') return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

export function buildForecastPoints(
  rows: Record<string, unknown>[],
  timeColumn: string,
  targetColumn: string,
): ForecastPoint[] {
  const points: ForecastPoint[] = []
  for (const row of rows) {
    const actual = asFiniteNumber(row[targetColumn])
    const predicted = asFiniteNumber(row['predicted'])
    if (actual === null || predicted === null) continue
    points.push({ date: String(row[timeColumn] ?? ''), actual, predicted })
  }
  return points
}

/**
 * Append future points as a dashed `forecast` series. The last historical
 * point also gets a forecast value so the dashed line connects seamlessly.
 */
export function appendFuture(
  history: ForecastPoint[],
  future: ForecastResponse | undefined,
): ForecastPoint[] {
  if (!future || future.points.length === 0) return history
  const merged = history.map((point, index) =>
    index === history.length - 1 ? { ...point, forecast: point.predicted } : point,
  )
  for (const point of future.points) {
    merged.push({
      date: point.date,
      forecast: point.predicted,
      band:
        point.low !== null && point.high !== null
          ? [point.low, point.high]
          : undefined,
    })
  }
  return merged
}
