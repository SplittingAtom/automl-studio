import { describe, expect, it } from 'vitest'

import { buildForecastPoints } from '../features/results/forecastData'

const ROWS = [
  { day: '2012-08-07', rentals: 7273, predicted: 7013.2, error: -259.8 },
  { day: '2012-08-08', rentals: 7534, predicted: 7273.9, error: -260.1 },
  { day: '2012-08-09', rentals: null, predicted: 7100.0, error: null },
  { day: '2012-08-10', rentals: 5786, predicted: 'oops', error: null },
]

describe('buildForecastPoints', () => {
  it('maps date, actual, and predicted from validation rows', () => {
    const points = buildForecastPoints(ROWS, 'day', 'rentals')
    expect(points[0]).toEqual({ date: '2012-08-07', actual: 7273, predicted: 7013.2 })
  })

  it('drops rows where actual or predicted is not numeric', () => {
    const points = buildForecastPoints(ROWS, 'day', 'rentals')
    expect(points).toHaveLength(2)
    expect(points.map((p) => p.date)).toEqual(['2012-08-07', '2012-08-08'])
  })
})
