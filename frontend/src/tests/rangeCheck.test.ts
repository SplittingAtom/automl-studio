import { describe, expect, it } from 'vitest'

import type { InputSpecItem } from '../api/schemas'
import { outOfRangeInputs } from '../features/whatif/rangeCheck'

const SPEC: InputSpecItem[] = [
  { name: 'age', kind: 'numeric', min_value: 0.4, max_value: 80, options: null, default: 28 },
  { name: 'fare', kind: 'numeric', min_value: 0, max_value: 512, options: null, default: 14 },
  { name: 'sex', kind: 'categorical', min_value: null, max_value: null, options: ['male', 'female'], default: 'male' },
]

describe('outOfRangeInputs', () => {
  it('returns nothing when all values are inside the observed range', () => {
    expect(outOfRangeInputs(SPEC, { age: 30, fare: 100, sex: 'female' })).toEqual([])
  })

  it('flags values above the observed maximum', () => {
    expect(outOfRangeInputs(SPEC, { age: 83, fare: 100 })).toEqual(['age'])
  })

  it('flags values below the observed minimum', () => {
    expect(outOfRangeInputs(SPEC, { age: 0.1, fare: -5 })).toEqual(['age', 'fare'])
  })

  it('boundary values are in range', () => {
    expect(outOfRangeInputs(SPEC, { age: 80, fare: 0 })).toEqual([])
  })

  it('ignores categorical and missing values', () => {
    expect(outOfRangeInputs(SPEC, { sex: 'male' })).toEqual([])
  })
})
