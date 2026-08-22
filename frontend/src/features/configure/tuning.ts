import type { TuningOverridesInput } from '../../api/schemas'

export type KnobKey =
  | 'max_depth'
  | 'learning_rate'
  | 'n_estimators'
  | 'subsample'
  | 'colsample_bytree'
  | 'min_child_weight'
  | 'reg_alpha'
  | 'reg_lambda'

export type MonotoneDirection = 1 | -1

export interface TuningState {
  params: Partial<Record<KnobKey, number>>
  monotone: Record<string, MonotoneDirection>
}

export const EMPTY_TUNING: TuningState = { params: {}, monotone: {} }

/** Ranges mirror the backend's TuningOverrides validation exactly. */
export const KNOBS: {
  key: KnobKey
  label: string
  min: number
  max: number
  step: number
  defaultValue: number
  hint: string
}[] = [
  {
    key: 'max_depth',
    label: 'Tree depth',
    min: 2,
    max: 12,
    step: 1,
    defaultValue: 6,
    hint: 'Deeper trees find more complex patterns but memorize more.',
  },
  {
    key: 'learning_rate',
    label: 'Learning speed',
    min: 0.005,
    max: 0.5,
    step: 0.005,
    defaultValue: 0.05,
    hint: 'Slower learning generalizes better but needs more rounds.',
  },
  {
    key: 'n_estimators',
    label: 'Training rounds',
    min: 50,
    max: 2000,
    step: 50,
    defaultValue: 500,
    hint: 'An upper limit — early stopping trims unused rounds.',
  },
  {
    key: 'subsample',
    label: 'Row sampling',
    min: 0.5,
    max: 1,
    step: 0.05,
    defaultValue: 1,
    hint: 'Train each round on a random share of rows to reduce memorizing.',
  },
  {
    key: 'colsample_bytree',
    label: 'Column sampling',
    min: 0.3,
    max: 1,
    step: 0.05,
    defaultValue: 1,
    hint: 'Give each tree a random share of columns.',
  },
  {
    key: 'min_child_weight',
    label: 'Minimum group size',
    min: 1,
    max: 32,
    step: 1,
    defaultValue: 1,
    hint: 'Higher values stop the model carving out tiny special cases.',
  },
  {
    key: 'reg_alpha',
    label: 'Simplicity pressure (L1)',
    min: 0,
    max: 10,
    step: 0.1,
    defaultValue: 0,
    hint: 'Pushes the model to ignore weak signals entirely.',
  },
  {
    key: 'reg_lambda',
    label: 'Simplicity pressure (L2)',
    min: 0,
    max: 10,
    step: 0.1,
    defaultValue: 1,
    hint: 'Gently shrinks every signal toward zero.',
  },
]

/** Convert panel state to the request payload; undefined when nothing is set. */
export function toOverrides(tuning: TuningState): TuningOverridesInput | undefined {
  const params = Object.fromEntries(
    Object.entries(tuning.params).filter(([, value]) => value !== undefined),
  )
  const hasMonotone = Object.keys(tuning.monotone).length > 0
  if (Object.keys(params).length === 0 && !hasMonotone) return undefined
  return {
    ...params,
    ...(hasMonotone ? { monotone_constraints: tuning.monotone } : {}),
  }
}
