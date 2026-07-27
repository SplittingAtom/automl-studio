import { statusMessage } from '../../lib/formatters'

export function TrainingProgress({ status }: { status: string }) {
  return (
    <div className="card" style={{ maxWidth: 480, margin: '4rem auto', textAlign: 'center' }}>
      <div className="spinner" />
      <h2>Building your model</h2>
      <p className="muted">{statusMessage(status)}</p>
      <p className="muted small">This usually takes a few seconds.</p>
    </div>
  )
}
