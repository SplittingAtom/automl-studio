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
  source: z.enum(['upload', 'sample']),
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

export const ModelMetaSchema = z.object({
  id: z.string(),
  dataset_id: z.string(),
  dataset_name: z.string(),
  target_column: z.string(),
  task: TaskSchema,
  status: ModelStatusSchema,
  created_at: z.string(),
  error: z.string().nullable(),
  metrics: z.record(z.string(), z.unknown()).nullable(),
  importance: z.array(z.object({ feature: z.string(), score: z.number() })).nullable(),
  input_spec: z.array(InputSpecItemSchema).nullable(),
  excluded_columns: z.array(z.object({ name: z.string(), reason: z.string() })),
  warnings: z.array(WarningSchema),
  n_rows_used: z.number().nullable(),
})

export const PredictResponseSchema = z.object({
  prediction: z.union([z.number(), z.string()]),
  probabilities: z
    .array(z.object({ label: z.string(), probability: z.number() }))
    .nullable(),
  elapsed_ms: z.number(),
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
export type WhatIfValues = Record<string, number | string>
