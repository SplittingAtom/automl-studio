import {
  keepPreviousData,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query'

import { api } from './client'
import type { Effort, ModelMeta, Task, WhatIfValues } from './schemas'

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
    }) => api.createModel(input),
  })
}

export function useModels(datasetId: string) {
  return useQuery({
    queryKey: ['models', 'list', datasetId],
    queryFn: () => api.listModels(datasetId),
  })
}

export function useValidationRows(modelId: string, enabled: boolean) {
  return useQuery({
    queryKey: ['models', modelId, 'validation'],
    queryFn: () => api.getValidationRows(modelId),
    enabled,
    retry: false, // older models have no saved validation; don't hammer the 404
  })
}

function isSettled(model: ModelMeta | undefined): boolean {
  return model?.status === 'complete' || model?.status === 'failed'
}

/** Polls every second until training settles. */
export function useModel(id: string) {
  return useQuery({
    queryKey: ['models', id],
    queryFn: () => api.getModel(id),
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
