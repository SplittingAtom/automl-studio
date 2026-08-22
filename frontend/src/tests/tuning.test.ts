import { describe, expect, it } from 'vitest'

import { EMPTY_TUNING, KNOBS, toOverrides } from '../features/configure/tuning'

describe('toOverrides', () => {
  it('returns undefined when nothing is set', () => {
    expect(toOverrides(EMPTY_TUNING)).toBeUndefined()
  })

  it('includes only the knobs the user set', () => {
    const overrides = toOverrides({
      params: { max_depth: 4, subsample: 0.8 },
      monotone: {},
    })
    expect(overrides).toEqual({ max_depth: 4, subsample: 0.8 })
  })

  it('includes direction rules when present', () => {
    const overrides = toOverrides({
      params: {},
      monotone: { sqft: 1, age: -1 },
    })
    expect(overrides).toEqual({ monotone_constraints: { sqft: 1, age: -1 } })
  })

  it('knob ranges contain their defaults', () => {
    for (const knob of KNOBS) {
      expect(knob.defaultValue).toBeGreaterThanOrEqual(knob.min)
      expect(knob.defaultValue).toBeLessThanOrEqual(knob.max)
    }
  })
})
