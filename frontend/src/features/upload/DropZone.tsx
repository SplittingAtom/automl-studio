import { useRef, useState, type DragEvent } from 'react'

export function DropZone({ onFile, busy }: { onFile: (file: File) => void; busy: boolean }) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragActive, setDragActive] = useState(false)

  const handleDrop = (event: DragEvent) => {
    event.preventDefault()
    setDragActive(false)
    const file = event.dataTransfer.files[0]
    if (file) onFile(file)
  }

  return (
    <div
      className={`dropzone${dragActive ? ' active' : ''}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(event) => {
        event.preventDefault()
        setDragActive(true)
      }}
      onDragLeave={() => setDragActive(false)}
      onDrop={handleDrop}
      role="button"
      aria-label="Upload a data file"
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.xlsx"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0]
          if (file) onFile(file)
          event.target.value = ''
        }}
      />
      {busy ? (
        <>
          <div className="spinner" />
          <strong>Reading your data…</strong>
        </>
      ) : (
        <>
          <strong>Drag a CSV or Excel file here</strong>
          <p className="muted small" style={{ marginBottom: 0 }}>
            or click to browse — .csv or .xlsx, up to 50 MB
          </p>
        </>
      )}
    </div>
  )
}
