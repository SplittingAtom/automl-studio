import type { InputSpecItem, WhatIfValues } from '../../api/schemas'

/**
 * Names of numeric inputs currently set beyond the range the model was
 * trained on. Tree models can't extrapolate — predictions out here plateau.
 */
export function outOfRangeInputs(spec: InputSpecItem[], values: WhatIfValues): string[] {
  return spec
    .filter((item) => item.kind === 'numeric' && item.min_value !== null && item.max_value !== null)
    .filter((item) => {
      const value = Number(values[item.name])
      return Number.isFinite(value) && (value < item.min_value! || value > item.max_value!)
    })
    .map((item) => item.name)
}
