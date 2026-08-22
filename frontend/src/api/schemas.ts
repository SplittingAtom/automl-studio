import { z } from 'zod'

export const EnvelopeSchema = z.object({
  success: z.boolean(),
  data: z.unknown(),
  error: z.object({ code: z.string(), message: z.string() }).nullable(),
  meta: z.record(z.string(), z.unknown()),
})

export const ColumnKindSchema = z.enum([
  'numeric',
  'categorical',
  'datetime',
  'id_like',
  'unsupported',
])

export const NumericStatsSchema = z.object({
  min: z.number(),
  max: z.number(),
  mean: z.number(),
  median: z.number(),
})

export const ColumnProfileSchema = z.object({
  name: z.string(),
  kind: ColumnKindSchema,
  dtype: z.string(),
  missing_count: z.number(),
  missing_pct: z.number(),
  unique_count: z.number(),
  stats: NumericStatsSchema.nullable(),
  top_values: z.array(z.object({ value: z.string(), count: z.number() })).nullable(),
})

export const WarningSchema = z.object({
  code: z.string(),
  message: z.string(),
  column: z.string().nullable(),
})

export const DatasetMetaSchema = z.object({
  id: z.string(),
  name: z.string(),
  source: z.enum(['upload', 'sample', 'derived']),
  created_at: z.string(),
  row_count: z.number(),
  column_count: z.number(),
  columns: z.array(ColumnProfileSchema),
  warnings: z.array(WarningSchema),
})

export const DatasetPreviewSchema = z.object({
  columns: z.array(z.string()),
  rows: z.array(z.record(z.string(), z.unknown())),
})

export const DistributionBinSchema = z.object({
  label: z.string(),
  count: z.number(),
  low: z.number().nullable(),
  high: z.number().nullable(),
})

export const ColumnExplorationSchema = z.object({
  name: z.string(),
  kind: ColumnKindSchema,
  missing_pct: z.number(),
  unique_count: z.number(),
  bins: z.array(DistributionBinSchema),
  other_count: z.number(),
  stats: z
    .object({
      min: z.number(),
      max: z.number(),
      mean: z.number(),
      median: z.number(),
      std: z.number(),
      outlier_count: z.number(),
    })
    .nullable(),
  note: z.string().nullable(),
})

export const ExplorationHighlightSchema = z.object({
  tone: z.enum(['info', 'warn']),
  message: z.string(),
  column: z.string().nullable(),
})

export const DatasetExplorationSchema = z.object({
  dataset_id: z.string(),
  row_count: z.number(),
  column_count: z.number(),
  missing_cells_pct: z.number(),
  duplicate_rows: z.number(),
  columns: z.array(ColumnExplorationSchema),
  highlights: z.array(ExplorationHighlightSchema),
  version: z.number(),
})

export const TaskSchema = z.enum(['classification', 'regression'])
export const ModelStatusSchema = z.enum(['queued', 'training', 'complete', 'failed'])

export const InputSpecItemSchema = z.object({
  name: z.string(),
  kind: z.enum(['numeric', 'categorical']),
  min_value: z.number().nullable(),
  max_value: z.number().nullable(),
  options: z.array(z.string()).nullable(),
  default: z.union([z.number(), z.string()]).nullable(),
})

export const TuningOverridesSchema = z.object({
  max_depth: z.number().nullable(),
  learning_rate: z.number().nullable(),
  n_estimators: z.number().nullable(),
  subsample: z.number().nullable(),
  colsample_bytree: z.number().nullable(),
  min_child_weight: z.number().nullable(),
  reg_alpha: z.number().nullable(),
  reg_lambda: z.number().nullable(),
  monotone_constraints: z.record(z.string(), z.number()),
})

/** Request-side shape: only send the knobs the user actually set. */
export type TuningOverridesInput = {
  max_depth?: number
  learning_rate?: number
  n_estimators?: number
  subsample?: number
  colsample_bytree?: number
  min_child_weight?: number
  reg_alpha?: number
  reg_lambda?: number
  monotone_constraints?: Record<string, 1 | -1>
}

export const ModelMetaSchema = z.object({
  id: z.string(),
  dataset_id: z.string(),
  dataset_name: z.string(),
  target_column: z.string(),
  task: TaskSchema,
  status: ModelStatusSchema,
  effort: z.enum(['standard', 'thorough']),
  time_column: z.string().nullable(),
  horizon: z.number(),
  created_at: z.string(),
  error: z.string().nullable(),
  metrics: z.record(z.string(), z.unknown()).nullable(),
  importance: z.array(z.object({ feature: z.string(), score: z.number() })).nullable(),
  input_spec: z.array(InputSpecItemSchema).nullable(),
  excluded_columns: z.array(z.object({ name: z.string(), reason: z.string() })),
  user_excluded_columns: z.array(z.string()),
  suggested_exclusions: z.array(z.string()),
  leak_suspect: z.string().nullable(),
  warnings: z.array(WarningSchema),
  n_rows_used: z.number().nullable(),
  overrides: TuningOverridesSchema.nullable(),
  baseline_model_id: z.string().nullable(),
  label: z.string().nullable(),
})

export const ExplanationSchema = z.object({
  items: z.array(z.object({ feature: z.string(), contribution: z.number() })),
  baseline: z.number(),
  toward_label: z.string().nullable(),
})

export const PredictResponseSchema = z.object({
  prediction: z.union([z.number(), z.string()]),
  probabilities: z
    .array(z.object({ label: z.string(), probability: z.number() }))
    .nullable(),
  explanation: ExplanationSchema.nullable(),
  interval: z.object({ low: z.number(), high: z.number() }).nullable(),
  elapsed_ms: z.number(),
})

export const ValidationRowsSchema = z.object({
  columns: z.array(z.string()),
  rows: z.array(z.record(z.string(), z.unknown())),
  total_rows: z.number(),
})

export const ForecastResponseSchema = z.object({
  points: z.array(
    z.object({
      date: z.string(),
      predicted: z.number(),
      low: z.number().nullable(),
      high: z.number().nullable(),
    }),
  ),
  last_actual_date: z.string(),
})

export const ThresholdPointSchema = z.object({
  threshold: z.number(),
  precision: z.number().nullable(),
  recall: z.number(),
  accuracy: z.number(),
  flagged_pct: z.number(),
})

export const SensitivityResponseSchema = z.object({
  feature: z.string(),
  kind: z.enum(['numeric', 'categorical']),
  output_label: z.string(),
  current_value: z.union([z.number(), z.string()]).nullable(),
  points: z.array(
    z.object({ value: z.union([z.number(), z.string()]), output: z.number() }),
  ),
})

export const CorrelationCellSchema = z.object({
  value: z.number().nullable(),
  signed: z.boolean(),
})

export const FeatureImpactSchema = z.object({
  feature: z.string(),
  kind: z.enum(['numeric', 'categorical']),
  mean_abs_contribution: z.number(),
  points: z.array(
    z.object({
      contribution: z.number(),
      value_norm: z.number().nullable(),
      value_label: z.string(),
    }),
  ),
})

export const InsightsResponseSchema = z.object({
  columns: z.array(z.string()),
  prediction_label: z.string(),
  matrix: z.array(z.array(CorrelationCellSchema)),
  impacts: z.array(FeatureImpactSchema),
  axis_low_label: z.string(),
  axis_high_label: z.string(),
  sample_size: z.number(),
  association_rows: z.number(),
})

export type BlueprintNode = {
  samples: number
  question: string | null
  label: string | null
  yes: BlueprintNode | null
  no: BlueprintNode | null
}

export const BlueprintNodeSchema: z.ZodType<BlueprintNode> = z.lazy(() =>
  z.object({
    samples: z.number(),
    question: z.string().nullable(),
    label: z.string().nullable(),
    yes: BlueprintNodeSchema.nullable(),
    no: BlueprintNodeSchema.nullable(),
  }),
)

export const BlueprintResponseSchema = z.object({
  root: BlueprintNodeSchema,
  fidelity: z.number(),
  output_label: z.string(),
  sample_size: z.number(),
})

export const GroupCheckResponseSchema = z.object({
  flagged: z.array(
    z.object({
      column: z.string(),
      value: z.string(),
      rows: z.number(),
      gap_label: z.string(),
    }),
  ),
  groups_checked: z.number(),
  overall_label: z.string(),
})

export const FeatureIdeaSchema = z.object({
  name: z.string(),
  formula: z.string(),
  share: z.number(),
  based_on: z.tuple([z.string(), z.string()]),
})

export const FeatureIdeasResponseSchema = z.object({
  ideas: z.array(FeatureIdeaSchema),
  checked: z.number(),
})

export const TargetCandidateSchema = z.object({
  column: z.string(),
  task: z.enum(['classification', 'regression']),
  score: z.number(),
  recommended: z.boolean(),
  derived_like: z.boolean(),
  signal: z.number(),
  probe_score: z.number().nullable(),
  baseline_score: z.number().nullable(),
  reasons: z.array(z.string()),
  top_predictors: z.array(z.object({ name: z.string(), share: z.number() })),
})

export const DatasetAnalysisSchema = z.object({
  dataset_id: z.string(),
  rating: z.enum(['great', 'good', 'fair', 'poor']),
  summary: z.string(),
  points: z.array(
    z.object({ tone: z.enum(['good', 'warn', 'bad']), message: z.string() }),
  ),
  candidates: z.array(TargetCandidateSchema),
})

export type ColumnKind = z.infer<typeof ColumnKindSchema>
export type ColumnProfile = z.infer<typeof ColumnProfileSchema>
export type ApiWarning = z.infer<typeof WarningSchema>
export type DatasetMeta = z.infer<typeof DatasetMetaSchema>
export type DatasetPreview = z.infer<typeof DatasetPreviewSchema>
export type Task = z.infer<typeof TaskSchema>
export type ModelStatus = z.infer<typeof ModelStatusSchema>
export type InputSpecItem = z.infer<typeof InputSpecItemSchema>
export type ModelMeta = z.infer<typeof ModelMetaSchema>
export type PredictResponse = z.infer<typeof PredictResponseSchema>
export type Explanation = z.infer<typeof ExplanationSchema>
export type TargetCandidate = z.infer<typeof TargetCandidateSchema>
export type DatasetAnalysis = z.infer<typeof DatasetAnalysisSchema>
export type SensitivityResponse = z.infer<typeof SensitivityResponseSchema>
export type ValidationRows = z.infer<typeof ValidationRowsSchema>
export type ForecastResponse = z.infer<typeof ForecastResponseSchema>
export type ThresholdPoint = z.infer<typeof ThresholdPointSchema>
export type CorrelationCell = z.infer<typeof CorrelationCellSchema>
export type DistributionBin = z.infer<typeof DistributionBinSchema>
export type ColumnExploration = z.infer<typeof ColumnExplorationSchema>
export type DatasetExploration = z.infer<typeof DatasetExplorationSchema>
export type FeatureImpact = z.infer<typeof FeatureImpactSchema>
export type InsightsResponse = z.infer<typeof InsightsResponseSchema>
export type TuningOverrides = z.infer<typeof TuningOverridesSchema>
export type FeatureIdea = z.infer<typeof FeatureIdeaSchema>
export type BlueprintResponse = z.infer<typeof BlueprintResponseSchema>
export type GroupCheckResponse = z.infer<typeof GroupCheckResponseSchema>
export type FeatureIdeasResponse = z.infer<typeof FeatureIdeasResponseSchema>
export type Effort = 'standard' | 'thorough'
export type WhatIfValues = Record<string, number | string>
