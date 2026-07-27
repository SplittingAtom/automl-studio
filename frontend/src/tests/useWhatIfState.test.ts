import { act, renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { InputSpecItem } from '../api/schemas'
import { initialValues, useWhatIfState } from '../features/whatif/useWhatIfState'

const SPEC: InputSpecItem[] = [
  {
    name: 'age',
    kind: 'numeric',
    min_value: 0,
    max_value: 80,
    options: null,
    default: 28,
  },
  {
    name: 'sex',
    kind: 'categorical',
    min_value: null,
    max_value: null,
    options: ['male', 'female'],
    default: 'male',
  },
]

describe('initialValues', () => {
  it('builds a typical row from defaults', () => {
    expect(initialValues(SPEC)).toEqual({ age: 28, sex: 'male' })
  })

  it('skips features without a default', () => {
    const spec = [{ ...SPEC[0], default: null }]
    expect(initialValues(spec)).toEqual({})
  })
})

describe('useWhatIfState', () => {
  it('updates a single value immutably', () => {
    const { result } = renderHook(() => useWhatIfState(SPEC))
    const before = result.current.values
    act(() => result.current.setValue('age', 60))
    expect(result.current.values).toEqual({ age: 60, sex: 'male' })
    expect(before).toEqual({ age: 28, sex: 'male' })
  })

  it('resets back to defaults', () => {
    const { result } = renderHook(() => useWhatIfState(SPEC))
    act(() => result.current.setValue('sex', 'female'))
    act(() => result.current.reset())
    expect(result.current.values).toEqual({ age: 28, sex: 'male' })
  })
})
