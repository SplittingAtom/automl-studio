import { useReducer } from 'react'

import type { InputSpecItem, WhatIfValues } from '../../api/schemas'

type Action =
  | { type: 'set'; name: string; value: number | string }
  | { type: 'reset'; spec: InputSpecItem[] }

export function initialValues(spec: InputSpecItem[]): WhatIfValues {
  const values: WhatIfValues = {}
  for (const item of spec) {
    if (item.default !== null) values[item.name] = item.default
  }
  return values
}

function reducer(state: WhatIfValues, action: Action): WhatIfValues {
  switch (action.type) {
    case 'set':
      return { ...state, [action.name]: action.value }
    case 'reset':
      return initialValues(action.spec)
  }
}

/** Form state for the what-if panel, initialized to "a typical row". */
export function useWhatIfState(spec: InputSpecItem[]) {
  const [values, dispatch] = useReducer(reducer, spec, initialValues)
  return {
    values,
    setValue: (name: string, value: number | string) => dispatch({ type: 'set', name, value }),
    reset: () => dispatch({ type: 'reset', spec }),
  }
}
