import type { ApiWarning } from '../api/schemas'

export function WarningBanner({ warnings }: { warnings: ApiWarning[] }) {
  if (warnings.length === 0) return null
  return (
    <div className="banner banner-warning" role="alert">
      <ul style={{ margin: 0, paddingLeft: '1.1rem' }}>
        {warnings.map((warning, index) => (
          <li key={`${warning.code}-${warning.column ?? index}`}>{warning.message}</li>
        ))}
      </ul>
    </div>
  )
}
