export function ErrorBanner({ error }: { error: unknown }) {
  if (!error) return null
  const message =
    error instanceof Error ? error.message : 'Something went wrong. Please try again.'
  return (
    <div className="banner banner-error" role="alert">
      {message}
    </div>
  )
}
