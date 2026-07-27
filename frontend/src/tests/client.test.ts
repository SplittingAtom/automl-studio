import { afterEach, describe, expect, it, vi } from 'vitest'

import { api, ApiError } from '../api/client'

function mockFetch(body: unknown, status = 200) {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      status,
      json: () => Promise.resolve(body),
    }),
  )
}

const DATASET = {
  id: 'ds_1',
  name: 'test.csv',
  source: 'upload',
  created_at: '2026-07-26T00:00:00Z',
  row_count: 10,
  column_count: 1,
  columns: [
    {
      name: 'age',
      kind: 'numeric',
      dtype: 'float64',
      missing_count: 0,
      missing_pct: 0,
      unique_count: 10,
      stats: { min: 1, max: 10, mean: 5, median: 5 },
      top_values: null,
    },
  ],
  warnings: [],
}

afterEach(() => vi.unstubAllGlobals())

describe('api client', () => {
  it('unwraps the envelope and validates the payload', async () => {
    mockFetch({ success: true, data: DATASET, error: null, meta: {} })
    const dataset = await api.getDataset('ds_1')
    expect(dataset.name).toBe('test.csv')
    expect(dataset.columns[0].stats?.median).toBe(5)
  })

  it('throws a typed error with the backend code and message', async () => {
    mockFetch(
      {
        success: false,
        data: null,
        error: { code: 'DATASET_NOT_FOUND', message: 'Gone.' },
        meta: {},
      },
      404,
    )
    const failure = api.getDataset('ds_x')
    await expect(failure).rejects.toBeInstanceOf(ApiError)
    await expect(failure).rejects.toMatchObject({ code: 'DATASET_NOT_FOUND', status: 404 })
  })

  it('rejects payloads that do not match the schema', async () => {
    mockFetch({ success: true, data: { nonsense: true }, error: null, meta: {} })
    await expect(api.getDataset('ds_1')).rejects.toMatchObject({ code: 'BAD_RESPONSE' })
  })

  it('handles non-JSON responses gracefully', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ status: 502, json: () => Promise.reject(new Error('nope')) }),
    )
    await expect(api.getDataset('ds_1')).rejects.toMatchObject({ code: 'BAD_RESPONSE' })
  })
})
