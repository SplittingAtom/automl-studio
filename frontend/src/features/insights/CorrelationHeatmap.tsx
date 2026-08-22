import { Fragment } from 'react'

import type { InsightsResponse } from '../../api/schemas'
import { cellColors, describeCell, strengthColor } from './color'

/**
 * Grid heatmap of how strongly each model input relates to the others and to
 * the prediction. Color carries strength (one hue, light → dark); the cell
 * text keeps the sign for numeric pairs, and the tooltip spells it out.
 */
export function CorrelationHeatmap({ insights }: { insights: InsightsResponse }) {
  const axes = [...insights.columns, insights.prediction_label]
  const last = axes.length - 1

  return (
    <div className="heatmap-scroll">
      <div
        className="heatmap"
        role="table"
        style={{
          gridTemplateColumns: `minmax(80px, max-content) repeat(${axes.length}, minmax(38px, 1fr))`,
        }}
      >
        <div />
        {axes.map((name, j) => (
          <div
            key={name}
            className={`heatmap-label heatmap-col-label${j === last ? ' prediction' : ''}`}
            title={name}
          >
            {name}
          </div>
        ))}
        {insights.matrix.map((row, i) => (
          <Fragment key={axes[i]}>
            <div
              className={`heatmap-label heatmap-row-label${i === last ? ' prediction' : ''}`}
              title={axes[i]}
            >
              {axes[i]}
            </div>
            {row.map((cell, j) => {
              if (cell.value === null) {
                return (
                  <div
                    key={axes[j]}
                    className="heatmap-cell empty"
                    title={`${axes[i]} ↔ ${axes[j]}: ${describeCell(cell)}`}
                  >
                    –
                  </div>
                )
              }
              const { background, ink } = cellColors(Math.abs(cell.value))
              return (
                <div
                  key={axes[j]}
                  className="heatmap-cell"
                  style={{ background, color: ink }}
                  title={`${axes[i]} ↔ ${axes[j]}: ${describeCell(cell)}`}
                >
                  {cell.value.toFixed(2)}
                </div>
              )
            })}
          </Fragment>
        ))}
      </div>
      <div className="heatmap-legend">
        <span className="muted small">not related</span>
        <div
          className="legend-gradient"
          style={{
            background: `linear-gradient(to right, ${strengthColor(0)}, ${strengthColor(0.5)}, ${strengthColor(1)})`,
          }}
        />
        <span className="muted small">strongly related</span>
      </div>
    </div>
  )
}
