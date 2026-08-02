export interface ForecastPoint {
  date: string
  actual: number
  predicted: number
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
