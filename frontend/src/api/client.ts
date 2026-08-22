import type { z } from 'zod'

import {
  DatasetAnalysisSchema,
  type DatasetExploration,
  DatasetExplorationSchema,
  DatasetMetaSchema,
  DatasetPreviewSchema,
  EnvelopeSchema,
  type GroupCheckResponse,
  GroupCheckResponseSchema,
  type BlueprintResponse,
  BlueprintResponseSchema,
  type FeatureIdeasResponse,
  FeatureIdeasResponseSchema,
  ModelMetaSchema,
  PredictResponseSchema,
  SensitivityResponseSchema,
  type DatasetAnalysis,
  type DatasetMeta,
  type DatasetPreview,
  type ModelMeta,
  type PredictResponse,
  type Effort,
  type ForecastResponse,
  ForecastResponseSchema,
  type InsightsResponse,
  InsightsResponseSchema,
  type SensitivityResponse,
  type Task,
  type TuningOverridesInput,
  type ValidationRows,
  type WhatIfValues,
  ValidationRowsSchema,
} from './schemas'

export class ApiError extends Error {
  readonly code: string
  readonly status: number

  constructor(code: string, message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
  }
}

async function request<T>(
  schema: z.ZodType<T>,
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(path, init)
  let body: unknown
  try {
    body = await response.json()
  } catch {
    throw new ApiError(
      'BAD_RESPONSE',
      'The server returned an unexpected response. Is the backend running?',
      response.status,
    )
  }
  const envelope = EnvelopeSchema.safeParse(body)
  if (!envelope.success) {
    throw new ApiError('BAD_RESPONSE', 'The server response had an unexpected shape.', response.status)
  }
  if (!envelope.data.success || envelope.data.error) {
    const error = envelope.data.error
    throw new ApiError(
      error?.code ?? 'UNKNOWN_ERROR',
      error?.message ?? 'Something went wrong.',
      response.status,
    )
  }
  const parsed = schema.safeParse(envelope.data.data)
  if (!parsed.success) {
    throw new ApiError('BAD_RESPONSE', 'The server data did not match the expected format.', response.status)
  }
  return parsed.data
}

export const api = {
  listDatasets: (): Promise<DatasetMeta[]> =>
    request(DatasetMetaSchema.array(), '/api/datasets'),

  getDataset: (id: string): Promise<DatasetMeta> =>
    request(DatasetMetaSchema, `/api/datasets/${id}`),

  previewDataset: (id: string, rows = 15): Promise<DatasetPreview> =>
    request(DatasetPreviewSchema, `/api/datasets/${id}/preview?rows=${rows}`),

  analyzeDataset: (id: string): Promise<DatasetAnalysis> =>
    request(DatasetAnalysisSchema, `/api/datasets/${id}/analysis`),

  exploreDataset: (id: string): Promise<DatasetExploration> =>
    request(DatasetExplorationSchema, `/api/datasets/${id}/exploration`),

  createCalculatedColumn: (
    datasetId: string,
    input: { name: string; formula: string },
  ): Promise<DatasetMeta> =>
    request(DatasetMetaSchema, `/api/datasets/${datasetId}/calculated`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    }),

  uploadDataset: (file: File): Promise<DatasetMeta> => {
    const form = new FormData()
    form.append('file', file)
    return request(DatasetMetaSchema, '/api/datasets', { method: 'POST', body: form })
  },

  createModel: (input: {
    dataset_id: string
    target_column: string
    task?: Task
    excluded_columns?: string[]
    effort?: Effort
    time_column?: string | null
    horizon?: number
    overrides?: TuningOverridesInput
    baseline_model_id?: string
    label?: string
  }): Promise<ModelMeta> =>
    request(ModelMetaSchema, '/api/models', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    }),

  getModel: (id: string): Promise<ModelMeta> =>
    request(ModelMetaSchema, `/api/models/${id}`),

  listModels: (datasetId: string): Promise<ModelMeta[]> =>
    request(ModelMetaSchema.array(), `/api/models?dataset_id=${encodeURIComponent(datasetId)}`),

  getValidationRows: (modelId: string, rows = 15): Promise<ValidationRows> =>
    request(ValidationRowsSchema, `/api/models/${modelId}/validation?rows=${rows}`),

  getGroupCheck: (modelId: string): Promise<GroupCheckResponse> =>
    request(GroupCheckResponseSchema, `/api/models/${modelId}/group-check`),

  getBlueprint: (modelId: string): Promise<BlueprintResponse> =>
    request(BlueprintResponseSchema, `/api/models/${modelId}/blueprint`),

  getFeatureIdeas: (modelId: string): Promise<FeatureIdeasResponse> =>
    request(FeatureIdeasResponseSchema, `/api/models/${modelId}/feature-ideas`),

  getInsights: (modelId: string): Promise<InsightsResponse> =>
    request(InsightsResponseSchema, `/api/models/${modelId}/insights`),

  getForecast: (modelId: string, steps: number): Promise<ForecastResponse> =>
    request(ForecastResponseSchema, `/api/models/${modelId}/forecast?steps=${steps}`),

  predict: (modelId: string, inputs: WhatIfValues): Promise<PredictResponse> =>
    request(PredictResponseSchema, `/api/models/${modelId}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ inputs }),
    }),

  sensitivity: (
    modelId: string,
    feature: string,
    inputs: WhatIfValues,
  ): Promise<SensitivityResponse> =>
    request(SensitivityResponseSchema, `/api/models/${modelId}/sensitivity`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ feature, inputs }),
    }),
}
