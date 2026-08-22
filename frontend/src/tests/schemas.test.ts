import { describe, expect, it } from 'vitest'

import {
  DatasetAnalysisSchema,
  DatasetExplorationSchema,
  InsightsResponseSchema,
  PredictResponseSchema,
  SensitivityResponseSchema,
} from '../api/schemas'

describe('PredictResponseSchema', () => {
  it('parses a classification response with explanation', () => {
    const parsed = PredictResponseSchema.parse({
      prediction: 'survived',
      probabilities: [
        { label: 'died', probability: 0.2 },
        { label: 'survived', probability: 0.8 },
      ],
      explanation: {
        items: [
          { feature: 'sex', contribution: 1.4 },
          { feature: 'age', contribution: -0.3 },
        ],
        baseline: -0.5,
        toward_label: 'survived',
      },
      interval: null,
      elapsed_ms: 2.1,
    })
    expect(parsed.explanation?.items[0].feature).toBe('sex')
  })

  it('parses a regression response without probabilities', () => {
    const parsed = PredictResponseSchema.parse({
      prediction: 231400.5,
      probabilities: null,
      explanation: { items: [], baseline: 200000, toward_label: null },
      interval: { low: 180000, high: 285000 },
      elapsed_ms: 1.0,
    })
    expect(parsed.prediction).toBe(231400.5)
    expect(parsed.interval?.low).toBe(180000)
  })
})

describe('DatasetAnalysisSchema', () => {
  it('parses a full analysis payload', () => {
    const parsed = DatasetAnalysisSchema.parse({
      dataset_id: 'ds_1',
      rating: 'good',
      summary: 'Good fit for modeling — "survived" is a strong target.',
      points: [
        { tone: 'good', message: '891 rows — plenty to learn from.' },
        { tone: 'warn', message: 'Heavy missing values in: deck.' },
      ],
      candidates: [
        {
          column: 'survived',
          task: 'classification',
          score: 78,
          recommended: true,
          derived_like: false,
          signal: 0.45,
          probe_score: 0.79,
          baseline_score: 0.62,
          reasons: ['The other columns predict it well.'],
          top_predictors: [{ name: 'sex', share: 0.6 }],
        },
      ],
    })
    expect(parsed.candidates[0].recommended).toBe(true)
    expect(parsed.rating).toBe('good')
  })
})

describe('DatasetExplorationSchema', () => {
  it('parses numeric, categorical, and excluded columns', () => {
    const parsed = DatasetExplorationSchema.parse({
      dataset_id: 'ds_1',
      row_count: 891,
      column_count: 3,
      missing_cells_pct: 8.1,
      duplicate_rows: 0,
      highlights: [
        { tone: 'info', message: '"fare" is stretched by a few large values.', column: 'fare' },
      ],
      version: 2,
      columns: [
        {
          name: 'age',
          kind: 'numeric',
          missing_pct: 19.9,
          unique_count: 88,
          bins: [{ label: '0.42–8.4', count: 54, low: 0.42, high: 8.4 }],
          other_count: 0,
          stats: { min: 0.42, max: 80, mean: 29.7, median: 28, std: 14.5, outlier_count: 11 },
          note: null,
        },
        {
          name: 'sex',
          kind: 'categorical',
          missing_pct: 0,
          unique_count: 2,
          bins: [
            { label: 'male', count: 577, low: null, high: null },
            { label: 'female', count: 314, low: null, high: null },
          ],
          other_count: 0,
          stats: null,
          note: null,
        },
        {
          name: 'ticket_id',
          kind: 'id_like',
          missing_pct: 0,
          unique_count: 891,
          bins: [],
          other_count: 0,
          stats: null,
          note: 'Looks like an ID — models leave it out.',
        },
      ],
    })
    expect(parsed.columns).toHaveLength(3)
    expect(parsed.columns[0].stats?.outlier_count).toBe(11)
    expect(parsed.columns[2].bins).toHaveLength(0)
  })
})

describe('InsightsResponseSchema', () => {
  it('parses a full insights payload', () => {
    const parsed = InsightsResponseSchema.parse({
      columns: ['age', 'sex'],
      prediction_label: 'Chance of "1"',
      matrix: [
        [
          { value: 1.0, signed: true },
          { value: 0.19, signed: false },
          { value: -0.08, signed: true },
        ],
        [
          { value: 0.19, signed: false },
          { value: 1.0, signed: false },
          { value: 0.54, signed: false },
        ],
        [
          { value: -0.08, signed: true },
          { value: 0.54, signed: false },
          { value: null, signed: false },
        ],
      ],
      impacts: [
        {
          feature: 'sex',
          kind: 'categorical',
          mean_abs_contribution: 1.2,
          points: [
            { contribution: -1.4, value_norm: null, value_label: 'male' },
            { contribution: 1.1, value_norm: null, value_label: 'female' },
          ],
        },
        {
          feature: 'age',
          kind: 'numeric',
          mean_abs_contribution: 0.4,
          points: [{ contribution: 0.3, value_norm: 0.07, value_label: '6' }],
        },
      ],
      axis_low_label: 'toward "0"',
      axis_high_label: 'toward "1"',
      sample_size: 179,
      association_rows: 179,
    })
    expect(parsed.matrix).toHaveLength(3)
    expect(parsed.impacts[0].points[0].value_norm).toBeNull()
  })

  it('rejects a malformed matrix', () => {
    expect(() =>
      InsightsResponseSchema.parse({
        columns: ['a'],
        prediction_label: 'p',
        matrix: [[0.5]],
        impacts: [],
        axis_low_label: 'x',
        axis_high_label: 'y',
        sample_size: 1,
        association_rows: 1,
      }),
    ).toThrow()
  })
})

describe('SensitivityResponseSchema', () => {
  it('parses numeric and categorical points', () => {
    const parsed = SensitivityResponseSchema.parse({
      feature: 'age',
      kind: 'numeric',
      output_label: 'Chance of "survived"',
      current_value: 28,
      points: [
        { value: 0.4, output: 0.9 },
        { value: 80, output: 0.2 },
      ],
    })
    expect(parsed.points).toHaveLength(2)

    const categorical = SensitivityResponseSchema.parse({
      feature: 'sex',
      kind: 'categorical',
      output_label: 'Chance of "survived"',
      current_value: 'male',
      points: [{ value: 'female', output: 0.75 }],
    })
    expect(categorical.points[0].value).toBe('female')
  })

  it('rejects malformed payloads', () => {
    expect(() =>
      SensitivityResponseSchema.parse({ feature: 'age', points: 'nope' }),
    ).toThrow()
  })
})
