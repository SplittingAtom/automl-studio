import type { ModelMeta } from '../../api/schemas'

/** The comparable headline score for a run, or null while it has none. */
export function headlineScore(model: ModelMeta): number | null {
  const metrics = model.metrics ?? {}
  const key = model.task === 'classification' ? 'accuracy' : 'r2'
  const value = metrics[key]
  return typeof value === 'number' ? value : null
}

/** What to call a run when the user didn't name it. */
export function variantLabel(model: ModelMeta): string {
  if (model.label) return model.label
  if (model.overrides !== null) return 'Fine-tuned'
  if (model.effort === 'thorough') return 'Thorough search'
  return 'Standard'
}

/**
 * Leaderboard order: in-progress runs first (they're what the user is
 * waiting on), then completed runs best-score-first, failures last.
 */
export function rankModels(models: ModelMeta[]): ModelMeta[] {
  const bucket = (m: ModelMeta) =>
    m.status === 'queued' || m.status === 'training' ? 0 : m.status === 'complete' ? 1 : 2
  return [...models].sort((a, b) => {
    const byBucket = bucket(a) - bucket(b)
    if (byBucket !== 0) return byBucket
    const scoreA = headlineScore(a)
    const scoreB = headlineScore(b)
    if (scoreA !== null && scoreB !== null && scoreA !== scoreB) return scoreB - scoreA
    return b.created_at.localeCompare(a.created_at)
  })
}

/** Id of the best completed run for the given target, if two can be compared. */
export function bestModelId(models: ModelMeta[]): string | null {
  const scored = models
    .filter((m) => m.status === 'complete')
    .map((m) => ({ id: m.id, key: `${m.target_column}|${m.task}`, score: headlineScore(m) }))
    .filter((m): m is { id: string; key: string; score: number } => m.score !== null)
  if (scored.length < 2) return null
  const byKey = new Map<string, { id: string; score: number }[]>()
  for (const row of scored) {
    byKey.set(row.key, [...(byKey.get(row.key) ?? []), row])
  }
  const groups = [...byKey.values()].filter((g) => g.length >= 2)
  if (groups.length === 0) return null
  const largest = groups.sort((a, b) => b.length - a.length)[0]
  return largest.sort((a, b) => b.score - a.score)[0].id
}
