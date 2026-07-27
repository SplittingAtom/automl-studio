import { describe, expect, it } from 'vitest'

import {
  DatasetAnalysisSchema,
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
