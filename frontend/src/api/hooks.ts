import {
  keepPreviousData,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query'

import { ApiError, api } from './client'
import type { Effort, ModelMeta, Task, TuningOverridesInput, WhatIfValues } from './schemas'

const POLL_INTERVAL_MS = 1000

export function useDatasets() {
  return useQuery({ queryKey: ['datasets'], queryFn: api.listDatasets })
}

export function useDataset(id: string) {
  return useQuery({ queryKey: ['datasets', id], queryFn: () => api.getDataset(id) })
}

export function useDatasetPreview(id: string) {
  return useQuery({
    queryKey: ['datasets', id, 'preview'],
    queryFn: () => api.previewDataset(id),
  })
}

/** Server computes once and caches to disk; may take a few seconds first time. */
export function useDatasetAnalysis(id: string) {
  return useQuery({
    queryKey: ['datasets', id, 'analysis'],
    queryFn: () => api.analyzeDataset(id),
    staleTime: Infinity,
  })
}

/** Per-column distributions; server computes once and caches to disk. */
export function useDatasetExploration(id: string) {
  return useQuery({
    queryKey: ['datasets', id, 'exploration'],
    queryFn: () => api.exploreDataset(id),
    staleTime: Infinity,
  })
}

export function useUploadDataset() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: api.uploadDataset,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['datasets'] }),
  })
}

export function useCreateCalculatedColumn(datasetId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (input: { name: string; formula: string }) =>
      api.createCalculatedColumn(datasetId, input),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['datasets'] }),
  })
}

export function useCreateModel() {
  return useMutation({
    mutationFn: (input: {
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
    }) => api.createModel(input),
  })
}

export function useModels(datasetId: string) {
  return useQuery({
    queryKey: ['models', 'list', datasetId],
    queryFn: () => api.listModels(datasetId),
    // Keep the leaderboard live while any run is still training.
    refetchInterval: (query) =>
      (query.state.data ?? []).some(
        (m) => m.status === 'queued' || m.status === 'training',
      )
        ? POLL_INTERVAL_MS
        : false,
  })
}

export function useValidationRows(modelId: string, enabled: boolean, rows = 15) {
  return useQuery({
    queryKey: ['models', modelId, 'validation', rows],
    queryFn: () => api.getValidationRows(modelId, rows),
    enabled,
    retry: false, // older models have no saved validation; don't hammer the 404
  })
}

function isSettled(model: ModelMeta | undefined): boolean {
  return model?.status === 'complete' || model?.status === 'failed'
}

/** Polls every second until training settles. */
export function useModel(id: string, enabled = true) {
  return useQuery({
    queryKey: ['models', id],
    queryFn: () => api.getModel(id),
    enabled: enabled && id !== '',
    refetchInterval: (query) => (isSettled(query.state.data) ? false : POLL_INTERVAL_MS),
  })
}

/**
 * Live what-if prediction. Keyed on the input values so repeated slider
 * positions hit the query cache; previous result stays visible while updating.
 */
export function usePrediction(modelId: string, inputs: WhatIfValues, enabled: boolean) {
  return useQuery({
    queryKey: ['models', modelId, 'predict', inputs],
    queryFn: () => api.predict(modelId, inputs),
    enabled,
    placeholderData: keepPreviousData,
    staleTime: Infinity,
  })
}

/** Per-group reliability check; server computes once per model. */
export function useGroupCheck(modelId: string, enabled: boolean) {
  return useQuery({
    queryKey: ['models', modelId, 'group-check'],
    queryFn: () => api.getGroupCheck(modelId),
    enabled,
    staleTime: Infinity,
    retry: (failureCount, error) =>
      !(error instanceof ApiError && error.status === 404) && failureCount < 2,
  })
}

/** Surrogate flowchart; server computes once per model. 404 = no validation. */
export function useBlueprint(modelId: string, enabled: boolean) {
  return useQuery({
    queryKey: ['models', modelId, 'blueprint'],
    queryFn: () => api.getBlueprint(modelId),
    enabled,
    staleTime: Infinity,
    retry: (failureCount, error) =>
      !(error instanceof ApiError && error.status === 404) && failureCount < 2,
  })
}

/** Suggested calculated columns; server computes once per model. */
export function useFeatureIdeas(modelId: string, enabled: boolean) {
  return useQuery({
    queryKey: ['models', modelId, 'feature-ideas'],
    queryFn: () => api.getFeatureIdeas(modelId),
    enabled,
    staleTime: Infinity,
  })
}

/** Global explainability computed server-side from held-out validation rows. */
export function useInsights(modelId: string, enabled: boolean) {
  return useQuery({
    queryKey: ['models', modelId, 'insights'],
    queryFn: () => api.getInsights(modelId),
    enabled,
    staleTime: Infinity,
    // Older models have no saved validation — the 404 is expected, don't
    // hammer it. Transient failures still get the normal retries.
    retry: (failureCount, error) =>
      !(error instanceof ApiError && error.status === 404) && failureCount < 2,
  })
}

/** Recursive future forecast for time-aware regression models. */
export function useForecast(modelId: string, steps: number, enabled: boolean) {
  return useQuery({
    queryKey: ['models', modelId, 'forecast', steps],
    queryFn: () => api.getForecast(modelId, steps),
    enabled: enabled && steps > 0,
    placeholderData: keepPreviousData,
    staleTime: Infinity,
  })
}

/** Sensitivity curve for one feature under the current what-if scenario. */
export function useSensitivity(
  modelId: string,
  feature: string | null,
  inputs: WhatIfValues,
) {
  return useQuery({
    queryKey: ['models', modelId, 'sensitivity', feature, inputs],
    queryFn: () => api.sensitivity(modelId, feature!, inputs),
    enabled: feature !== null,
    placeholderData: keepPreviousData,
    staleTime: Infinity,
  })
}
