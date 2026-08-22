import { describe, expect, it } from 'vitest'

import type { ModelMeta } from '../api/schemas'
import {
  bestModelId,
  headlineScore,
  rankModels,
  variantLabel,
} from '../features/models/leaderboard'

function model(partial: Partial<ModelMeta>): ModelMeta {
  return {
    id: 'mdl_1',
    dataset_id: 'ds_1',
    dataset_name: 'Data',
    target_column: 'y',
    task: 'classification',
    status: 'complete',
    effort: 'standard',
    time_column: null,
    horizon: 0,
    created_at: '2026-08-12T10:00:00Z',
    error: null,
    metrics: { accuracy: 0.8 },
    importance: null,
    input_spec: null,
    excluded_columns: [],
    user_excluded_columns: [],
    suggested_exclusions: [],
    leak_suspect: null,
    warnings: [],
    n_rows_used: 100,
    overrides: null,
    baseline_model_id: null,
    label: null,
    ...partial,
  }
}

describe('variantLabel', () => {
  it('prefers the user label, then tuning, then effort', () => {
    expect(variantLabel(model({ label: 'My run' }))).toBe('My run')
    expect(
      variantLabel(
        model({
          overrides: {
            max_depth: 3, learning_rate: null, n_estimators: null, subsample: null,
            colsample_bytree: null, min_child_weight: null, reg_alpha: null,
            reg_lambda: null, monotone_constraints: {},
          },
        }),
      ),
    ).toBe('Fine-tuned')
    expect(variantLabel(model({ effort: 'thorough' }))).toBe('Thorough search')
    expect(variantLabel(model({}))).toBe('Standard')
  })
})

describe('rankModels', () => {
  it('puts running first, then best scores, then failures', () => {
    const ranked = rankModels([
      model({ id: 'worse', metrics: { accuracy: 0.7 } }),
      model({ id: 'failed', status: 'failed', metrics: null }),
      model({ id: 'best', metrics: { accuracy: 0.9 } }),
      model({ id: 'running', status: 'training', metrics: null }),
    ])
    expect(ranked.map((m) => m.id)).toEqual(['running', 'best', 'worse', 'failed'])
  })
})

describe('bestModelId', () => {
  it('needs at least two comparable runs of the same target', () => {
    expect(bestModelId([model({})])).toBeNull()
    expect(
      bestModelId([
        model({ id: 'a', metrics: { accuracy: 0.8 } }),
        model({ id: 'b', target_column: 'other', metrics: { accuracy: 0.9 } }),
      ]),
    ).toBeNull()
  })

  it('picks the top scorer among comparable runs', () => {
    expect(
      bestModelId([
        model({ id: 'a', metrics: { accuracy: 0.8 } }),
        model({ id: 'b', metrics: { accuracy: 0.92 } }),
      ]),
    ).toBe('b')
  })
})

describe('headlineScore', () => {
  it('uses accuracy for classification and R² for regression', () => {
    expect(headlineScore(model({}))).toBe(0.8)
    expect(headlineScore(model({ task: 'regression', metrics: { r2: 0.6 } }))).toBe(0.6)
    expect(headlineScore(model({ metrics: null, status: 'training' }))).toBeNull()
  })
})
