import { useNavigate } from 'react-router-dom'

import { useDatasets, useUploadDataset } from '../../api/hooks'
import { ErrorBanner } from '../../components/ErrorBanner'
import { DropZone } from './DropZone'
import { SampleDatasetPicker } from './SampleDatasetPicker'

export function UploadPage() {
  const navigate = useNavigate()
  const datasets = useDatasets()
  const upload = useUploadDataset()

  const handleFile = (file: File) => {
    upload.mutate(file, {
      onSuccess: (dataset) => navigate(`/datasets/${dataset.id}/configure`),
    })
  }

  const samples = (datasets.data ?? []).filter((d) => d.source === 'sample')
  const uploads = (datasets.data ?? []).filter((d) => d.source !== 'sample')

  return (
    <div>
      <h1>Build a prediction model from your data</h1>
      <p className="muted">
        Point AutoML Studio at a spreadsheet, pick the column you want to predict, and
        explore the result — no machine learning experience needed.
      </p>

      <ErrorBanner error={upload.error} />
      <DropZone onFile={handleFile} busy={upload.isPending} />

      <h2 style={{ marginTop: '2rem' }}>Or try a sample dataset</h2>
      <SampleDatasetPicker
        samples={samples}
        loading={datasets.isLoading}
        error={datasets.error}
        onPick={(id) => navigate(`/datasets/${id}/configure`)}
      />

      {uploads.length > 0 && (
        <>
          <h2 style={{ marginTop: '2rem' }}>Your datasets</h2>
          <div className="sample-grid">
            {uploads.map((dataset) => (
              <button
                key={dataset.id}
                className="sample-card"
                onClick={() => navigate(`/datasets/${dataset.id}/configure`)}
              >
                <h3>{dataset.name}</h3>
                <span className="muted small">
                  {dataset.row_count.toLocaleString()} rows · {dataset.column_count} columns
                </span>
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
