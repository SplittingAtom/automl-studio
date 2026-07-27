import { describe, expect, it } from 'vitest'

import { formatCell, formatNumber, formatPercent, taskLabel } from '../lib/formatters'

describe('formatNumber', () => {
  it('drops decimals for large values', () => {
    expect(formatNumber(231400.5)).toBe('231,401')
  })

  it('keeps a decimal for mid-range values', () => {
    expect(formatNumber(123.45)).toBe('123.5')
  })

  it('keeps two decimals for small values', () => {
    expect(formatNumber(3.14159)).toBe('3.14')
  })
})

describe('formatPercent', () => {
  it('rounds ordinary fractions to whole percent', () => {
    expect(formatPercent(0.834)).toBe('83%')
  })

  it('keeps a decimal near 100% so 99.95% is not shown as 100%', () => {
    expect(formatPercent(0.9995)).toBe('100.0%')
  })
})

describe('formatCell', () => {
  it('renders empty values as an em dash', () => {
    expect(formatCell(null)).toBe('—')
    expect(formatCell(undefined)).toBe('—')
    expect(formatCell('')).toBe('—')
  })

  it('renders booleans and strings as text', () => {
    expect(formatCell(true)).toBe('true')
    expect(formatCell('male')).toBe('male')
  })
})

describe('taskLabel', () => {
  it('uses analyst language', () => {
    expect(taskLabel('classification')).toBe('Predict a category')
    expect(taskLabel('regression')).toBe('Predict a number')
  })
})
