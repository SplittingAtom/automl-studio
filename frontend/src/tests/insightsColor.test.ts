import { describe, expect, it } from 'vitest'

import {
  cellColors,
  describeCell,
  dotColor,
  jitter,
  strengthColor,
} from '../features/insights/color'

describe('strengthColor', () => {
  it('recedes toward the surface at zero and darkens toward one', () => {
    expect(strengthColor(0)).toBe('#f7f9fc')
    expect(strengthColor(1)).toBe('#0d366b')
  })

  it('clamps out-of-range input', () => {
    expect(strengthColor(-2)).toBe(strengthColor(0))
    expect(strengthColor(5)).toBe(strengthColor(1))
  })

  it('interpolates between stops', () => {
    expect(strengthColor(0.5)).toMatch(/^#[0-9a-f]{6}$/)
    expect(strengthColor(0.5)).not.toBe(strengthColor(0))
    expect(strengthColor(0.5)).not.toBe(strengthColor(1))
  })
})

describe('dotColor', () => {
  it('uses the muted gray for categories and missing values', () => {
    expect(dotColor(null)).toBe('#898781')
  })

  it('maps low and high values onto the blue ramp', () => {
    expect(dotColor(0)).toBe('#9ec5f4')
    expect(dotColor(1)).toBe('#0d366b')
  })
})

describe('cellColors', () => {
  it('flips to light ink on dark cells', () => {
    const weak = cellColors(0.1)
    const strong = cellColors(0.95)
    expect(weak.ink).not.toBe('#ffffff')
    expect(strong.ink).toBe('#ffffff')
  })
})

describe('describeCell', () => {
  it('explains signed correlations in plain English', () => {
    expect(describeCell({ value: -0.62, signed: true })).toContain('opposite')
    expect(describeCell({ value: 0.62, signed: true })).toContain('together')
  })

  it('explains strengths without inventing a direction', () => {
    const text = describeCell({ value: 0.4, signed: false })
    expect(text).toContain('strength')
    expect(text).not.toContain('opposite')
  })

  it('handles missing associations', () => {
    expect(describeCell({ value: null, signed: false })).toContain('Not enough')
  })
})

describe('jitter', () => {
  it('is deterministic and stays within a half row either way', () => {
    for (let i = 0; i < 50; i += 1) {
      expect(jitter(i)).toBe(jitter(i))
      expect(Math.abs(jitter(i))).toBeLessThanOrEqual(0.5)
    }
  })
})
