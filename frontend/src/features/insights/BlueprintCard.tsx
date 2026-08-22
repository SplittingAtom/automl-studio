import { useBlueprint } from '../../api/hooks'
import type { BlueprintNode, ModelMeta } from '../../api/schemas'

/**
 * "How the model decides": a shallow surrogate tree rendered as a
 * plain-English flowchart, with an honest fidelity disclaimer.
 */
export function BlueprintCard({ meta }: { meta: ModelMeta }) {
  const blueprint = useBlueprint(meta.id, meta.status === 'complete')
  if (blueprint.isLoading || blueprint.error || !blueprint.data) return null

  const data = blueprint.data
  const fidelityNote =
    meta.task === 'classification' && data.output_label.startsWith('Predicted')
      ? `agrees with the full model ${Math.round(data.fidelity * 100)}% of the time`
      : `captures about ${Math.round(data.fidelity * 100)}% of the full model's behavior`

  return (
    <div className="card">
      <h2>How the model decides — simplified</h2>
      <p className="muted small">
        A deliberately simple flowchart approximation of the model ({data.output_label.toLowerCase()}), built
        from {data.sample_size} held-out rows. It {fidelityNote}; the full model is
        more nuanced than any flowchart.
      </p>
      <div className="blueprint">
        <Node node={data.root} />
      </div>
    </div>
  )
}

function Node({ node }: { node: BlueprintNode }) {
  if (node.question === null) {
    return (
      <span>
        <strong>{node.label}</strong>{' '}
        <span className="muted small">({node.samples.toLocaleString()} rows)</span>
      </span>
    )
  }
  return (
    <div className="bp-node">
      <div className="bp-question">{node.question}</div>
      {node.yes && (
        <div className="bp-branch">
          <span className="bp-tag">yes</span>
          <Node node={node.yes} />
        </div>
      )}
      {node.no && (
        <div className="bp-branch">
          <span className="bp-tag bp-tag-no">no</span>
          <Node node={node.no} />
        </div>
      )}
    </div>
  )
}
