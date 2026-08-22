import { useGroupCheck } from '../../api/hooks'
import type { ModelMeta } from '../../api/schemas'

/**
 * "Where the model struggles": groups of rows served notably worse than
 * average — a plain-English reliability and fairness check.
 */
export function GroupCheckCard({ meta }: { meta: ModelMeta }) {
  const check = useGroupCheck(meta.id, meta.status === 'complete')
  if (check.isLoading || check.error || !check.data) return null
  const data = check.data
  if (data.groups_checked === 0) return null

  return (
    <div className="card">
      <h2>Where the model struggles</h2>
      <p className="muted small">
        Every group of at least 20 held-out rows was checked against the overall
        score ({data.overall_label}).
      </p>
      {data.flagged.length === 0 ? (
        <p className="small" style={{ color: 'var(--success)' }}>
          ✓ All {data.groups_checked} groups perform close to the overall score —
          no part of your data is being served notably worse.
        </p>
      ) : (
        data.flagged.map((flag) => (
          <div key={`${flag.column}-${flag.value}`} className="group-flag">
            <strong>
              {flag.column} = {flag.value}
            </strong>{' '}
            <span className="muted small">({flag.rows.toLocaleString()} rows)</span>
            <div className="small">{flag.gap_label}</div>
          </div>
        ))
      )}
    </div>
  )
}
