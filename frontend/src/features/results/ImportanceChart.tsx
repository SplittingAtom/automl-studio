import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const TOP_FEATURES = 10
const BAR_COLOR = '#4f46e5'

export function ImportanceChart({
  importance,
}: {
  importance: { feature: string; score: number }[]
}) {
  const data = importance.slice(0, TOP_FEATURES)
  if (data.length === 0) return <p className="muted">No feature importance available.</p>

  return (
    <ResponsiveContainer width="100%" height={Math.max(180, data.length * 36)}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24 }}>
        <CartesianGrid horizontal={false} strokeDasharray="3 3" />
        <XAxis
          type="number"
          tickFormatter={(value: number) => `${Math.round(value * 100)}%`}
          fontSize={12}
        />
        <YAxis type="category" dataKey="feature" width={120} fontSize={12} />
        <Tooltip
          formatter={(value) => [`${((value as number) * 100).toFixed(1)}%`, 'influence']}
        />
        <Bar dataKey="score" fill={BAR_COLOR} radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}
