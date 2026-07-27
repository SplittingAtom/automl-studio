import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { useCreateCalculatedColumn } from '../../api/hooks'
import type { ColumnProfile } from '../../api/schemas'
import { ErrorBanner } from '../../components/ErrorBanner'

const USABLE_KINDS = new Set(['numeric', 'categorical'])

/** Column names with anything beyond \w need pandas backticks in formulas. */
function formulaToken(name: string): string {
  return /^[A-Za-z_][A-Za-z0-9_]*$/.test(name) ? name : `\`${name}\``
}

export function CalculatedColumnCard({
  datasetId,
  columns,
}: {
  datasetId: string
  columns: ColumnProfile[]
}) {
  const navigate = useNavigate()
  const create = useCreateCalculatedColumn(datasetId)
  const [name, setName] = useState('')
  const [formula, setFormula] = useState('')
  const formulaRef = useRef<HTMLInputElement>(null)

  const insertColumn = (columnName: string) => {
    const token = formulaToken(columnName)
    setFormula((current) => (current ? `${current} ${token}` : token))
    formulaRef.current?.focus()
  }

  const handleCreate = () => {
    create.mutate(
      { name: name.trim(), formula: formula.trim() },
      { onSuccess: (derived) => navigate(`/datasets/${derived.id}/configure`) },
    )
  }

  const usable = columns.filter((c) => USABLE_KINDS.has(c.kind))

  return (
    <div className="card" style={{ marginTop: '1.25rem' }}>
      <h2>Add a calculated column</h2>
      <p className="muted small">
        Combine existing columns with a formula — like a calculated column in PowerBI.
        This creates a new copy of the dataset with your column added.
      </p>
      <div className="calc-form">
        <input
          type="text"
          className="calc-name"
          placeholder="new_column_name"
          aria-label="New column name"
          value={name}
          onChange={(event) => setName(event.target.value)}
        />
        <span className="muted">=</span>
        <input
          ref={formulaRef}
          type="text"
          className="calc-formula"
          placeholder="e.g. fare / (sibsp + parch + 1)"
          aria-label="Formula"
          value={formula}
          onChange={(event) => setFormula(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && name.trim() && formula.trim()) handleCreate()
          }}
        />
        <button
          className="btn btn-primary"
          disabled={!name.trim() || !formula.trim() || create.isPending}
          onClick={handleCreate}
        >
          {create.isPending ? 'Creating…' : 'Create column'}
        </button>
      </div>
      <div className="calc-chips">
        <span className="muted small">Insert a column:</span>
        {usable.map((column) => (
          <button
            key={column.name}
            className="chip chip-clickable"
            onClick={() => insertColumn(column.name)}
          >
            {column.name}
          </button>
        ))}
      </div>
      <p className="muted small" style={{ marginBottom: 0 }}>
        Supports + − × ÷ with parentheses, and comparisons like{' '}
        <code>fare &gt; 10</code> or <code>sex == 'male'</code>.
      </p>
      <ErrorBanner error={create.error} />
    </div>
  )
}
