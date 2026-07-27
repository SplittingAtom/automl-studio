import {
  keepPreviousData,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query'

import { api } from './client'
import type { ModelMeta, Task, WhatIfValues } from './schemas'

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

export function useUploadDataset() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: api.uploadDataset,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['datasets'] }),
  })
}

export function useCreateModel() {
  return useMutation({
    mutationFn: (input: { dataset_id: string; target_column: string; task?: Task }) =>
      api.createModel(input),
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
