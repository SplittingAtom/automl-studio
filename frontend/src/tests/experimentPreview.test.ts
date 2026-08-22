import { describe, expect, it } from 'vitest'

import { experimentPreview } from '../features/configure/experimentPreview'

const base = {
  rowCount: 1000,
  thorough: false,
  timeColumn: null,
  horizon: 0,
  tunedSettings: 0,
}

describe('experimentPreview', () => {
  it('describes a random split with the actual row count', () => {
    const lines = experimentPreview(base).join(' ')
    expect(lines).toContain('random 200 rows')
    expect(lines).toContain('one model with sensible defaults')
  })

  it('describes time-aware splits and the horizon gap', () => {
    const lines = experimentPreview({
      ...base,
      timeColumn: 'date',
      horizon: 7,
    }).join(' ')
    expect(lines).toContain('most recent 200 rows')
    expect(lines).toContain('ordered by date')
    expect(lines).toContain('7-row gap')
  })

  it('mentions the variation search only for thorough runs', () => {
    expect(experimentPreview({ ...base, thorough: true }).join(' ')).toContain(
      '12 model variations',
    )
    expect(experimentPreview(base).join(' ')).not.toContain('12 model variations')
  })

  it('skips early stopping for tiny datasets', () => {
    const lines = experimentPreview({ ...base, rowCount: 100 }).join(' ')
    expect(lines).not.toContain('stall')
  })

  it('mentions sampling only above the row cap', () => {
    expect(experimentPreview({ ...base, rowCount: 250_000 }).join(' ')).toContain(
      '100,000-row sample',
    )
    expect(experimentPreview(base).join(' ')).not.toContain('sample for speed')
  })

  it('counts pinned advanced settings', () => {
    expect(experimentPreview({ ...base, tunedSettings: 2 }).join(' ')).toContain(
      'Pins 2 advanced settings',
    )
  })
})
