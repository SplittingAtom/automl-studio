import { useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { useSensitivity } from '../../api/hooks'
import type { ModelMeta, SensitivityResponse, WhatIfValues } from '../../api/schemas'
import { ErrorBanner } from '../../components/ErrorBanner'
import { InfoTip } from '../../components/InfoTip'
import { formatNumber } from '../../lib/formatters'

const LINE_COLOR = '#4f46e5'

function defaultFeature(meta: ModelMeta): string | null {
  const specNames = new Set((meta.input_spec ?? []).map((i) => i.name))
  const topImportant = (meta.importance ?? []).find((i) => specNames.has(i.feature))
  return topImportant?.feature ?? meta.input_spec?.[0]?.name ?? null
}

export function SensitivityCard({
  meta,
  inputs,
}: {
  meta: ModelMeta
  inputs: WhatIfValues
}) {
  const [feature, setFeature] = useState<string | null>(() => defaultFeature(meta))
  const sensitivity = useSensitivity(meta.id, feature, inputs)

  if (!meta.input_spec || meta.input_spec.length === 0) return null

  return (
    <div className="card" style={{ marginTop: '1.25rem' }}>
      <div className="control-header">
        <h2 style={{ marginBottom: 0 }}>
          How does{' '}
          <select
            className="inline-select"
            aria-label="Sensitivity feature"
            value={feature ?? ''}
            onChange={(event) => setFeature(event.target.value)}
          >
            {meta.input_spec.map((item) => (
              <option key={item.name} value={item.name}>
                {item.name}
              </option>
            ))}
          </select>{' '}
          affect the prediction?
          <InfoTip tip="Sweeps this one column across its full range while every other input stays at your current what-if values. The curve shows this scenario's sensitivity — change the other inputs and the curve can change too." />
        </h2>
      </div>
      <p className="muted small">
        Everything else stays at the values set in the what-if panel.
      </p>
      <ErrorBanner error={sensitivity.error} />
      {sensitivity.data && <SensitivityChart data={sensitivity.data} />}
      {!sensitivity.data && !sensitivity.error && <p className="muted">Calculating…</p>}
    </div>
  )
}

function SensitivityChart({ data }: { data: SensitivityResponse }) {
  const points = data.points.map((p) => ({ ...p }))
  const isProbability = data.output_label.startsWith('Chance')
  const formatOutput = (value: number) =>
    isProbability ? `${Math.round(value * 100)}%` : formatNumber(value)

  if (data.kind === 'categorical') {
    return (
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={points} margin={{ left: 8, right: 24 }}>
          <CartesianGrid vertical={false} strokeDasharray="3 3" />
          <XAxis dataKey="value" fontSize={12} />
          <YAxis tickFormatter={formatOutput} fontSize={12} width={60} />
          <Tooltip formatter={(value) => [formatOutput(value as number), data.output_label]} />
          <Bar dataKey="output" fill={LINE_COLOR} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={points} margin={{ left: 8, right: 24 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="value"
          type="number"
          domain={['dataMin', 'dataMax']}
          tickFormatter={(value: number) => formatNumber(value)}
          fontSize={12}
        />
        <YAxis tickFormatter={formatOutput} fontSize={12} width={60} />
        <Tooltip
          labelFormatter={(value) => `${data.feature}: ${formatNumber(value as number)}`}
          formatter={(value) => [formatOutput(value as number), data.output_label]}
        />
        {typeof data.current_value === 'number' && (
          <ReferenceLine x={data.current_value} stroke="#94a3b8" strokeDasharray="4 4" />
        )}
        <Line type="monotone" dataKey="output" stroke={LINE_COLOR} strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}
