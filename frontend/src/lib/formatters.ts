import type { ColumnKind, Task } from '../api/schemas'

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return '—'
  const abs = Math.abs(value)
  if (abs >= 100000) return value.toLocaleString('en-US', { maximumFractionDigits: 0 })
  if (abs >= 100) return value.toLocaleString('en-US', { maximumFractionDigits: 1 })
  return value.toLocaleString('en-US', { maximumFractionDigits: 2 })
}

export function formatPercent(fraction: number): string {
  return `${(fraction * 100).toFixed(fraction >= 0.995 ? 1 : 0)}%`
}

export function formatCell(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'number') return formatNumber(value)
  return String(value)
}

export function taskLabel(task: Task): string {
  return task === 'classification' ? 'Predict a category' : 'Predict a number'
}

export function columnKindLabel(kind: ColumnKind): string {
  const labels: Record<ColumnKind, string> = {
    numeric: 'Number',
    categorical: 'Category',
    datetime: 'Date',
    id_like: 'ID',
    unsupported: 'Empty',
  }
  return labels[kind]
}

export function statusMessage(status: string): string {
  const messages: Record<string, string> = {
    queued: 'Getting things ready…',
    training: 'Teaching the model with your data…',
    complete: 'Done!',
    failed: 'Training failed.',
  }
  return messages[status] ?? status
}
