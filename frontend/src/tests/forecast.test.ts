import { describe, expect, it } from 'vitest'

import { appendFuture, buildForecastPoints } from '../features/results/forecastData'

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

describe('appendFuture', () => {
  const history = buildForecastPoints(ROWS, 'day', 'rentals')
  const future = {
    points: [
      { date: '2012-08-11', predicted: 6900, low: 6100, high: 7600 },
      { date: '2012-08-12', predicted: 6800, low: null, high: null },
    ],
    last_actual_date: '2012-08-10',
  }

  it('appends future points as the forecast series with bands', () => {
    const merged = appendFuture(history, future)
    expect(merged).toHaveLength(4)
    expect(merged[2]).toEqual({ date: '2012-08-11', forecast: 6900, band: [6100, 7600] })
    expect(merged[3].band).toBeUndefined()
  })

  it('bridges the dashed line from the last historical prediction', () => {
    const merged = appendFuture(history, future)
    expect(merged[1].forecast).toBe(merged[1].predicted)
  })

  it('returns history untouched when there is no forecast', () => {
    expect(appendFuture(history, undefined)).toEqual(history)
  })
})
