import type { CorrelationCell } from '../../api/schemas'

/**
 * Sequential blue ramp (light → dark) for relationship strength. The lightest
 * stop recedes toward the card surface so "no relationship" reads as nothing.
 */
const STRENGTH_STOPS = ['#f7f9fc', '#9ec5f4', '#3987e5', '#1c5cab', '#0d366b']

/** Dot fill for the beeswarm: starts darker so small marks stay visible. */
const DOT_STOPS = ['#9ec5f4', '#3987e5', '#0d366b']

/** Muted gray for dots whose column has no numeric "low → high" (categories). */
export const NEUTRAL_DOT = '#898781'

/** Ink flips to white once the cell background gets dark enough to swallow text. */
const LIGHT_INK_THRESHOLD = 0.55

function hexToRgb(hex: string): [number, number, number] {
  return [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16),
  ]
}

function rgbToHex([r, g, b]: [number, number, number]): string {
  const channel = (v: number) => Math.round(v).toString(16).padStart(2, '0')
  return `#${channel(r)}${channel(g)}${channel(b)}`
}

function ramp(stops: string[], t: number): string {
  const clamped = Math.min(1, Math.max(0, t))
  const scaled = clamped * (stops.length - 1)
  const index = Math.min(Math.floor(scaled), stops.length - 2)
  const local = scaled - index
  const from = hexToRgb(stops[index])
  const to = hexToRgb(stops[index + 1])
  return rgbToHex([
    from[0] + (to[0] - from[0]) * local,
    from[1] + (to[1] - from[1]) * local,
    from[2] + (to[2] - from[2]) * local,
  ])
}

/** Cell background for a relationship strength in 0..1. */
export function strengthColor(strength: number): string {
  return ramp(STRENGTH_STOPS, strength)
}

/** Beeswarm dot fill for a column value's 0..1 position (null = category/missing). */
export function dotColor(valueNorm: number | null): string {
  if (valueNorm === null) return NEUTRAL_DOT
  return ramp(DOT_STOPS, valueNorm)
}

/** Background + readable ink for a heatmap cell. */
export function cellColors(strength: number): { background: string; ink: string } {
  return {
    background: strengthColor(strength),
    ink: strength > LIGHT_INK_THRESHOLD ? '#ffffff' : 'var(--text)',
  }
}

/** Plain-English tooltip body for one heatmap cell. */
export function describeCell(cell: Pick<CorrelationCell, 'value' | 'signed'>): string {
  if (cell.value === null) return 'Not enough overlapping data to measure.'
  if (cell.signed) {
    const direction =
      cell.value >= 0 ? 'they tend to move together' : 'they move in opposite directions'
    return `Correlation ${cell.value.toFixed(2)} — ${direction}.`
  }
  return `Relationship strength ${cell.value.toFixed(2)} (0 = none, 1 = perfectly linked).`
}

/**
 * Deterministic vertical spread for beeswarm dots (golden-ratio sequence):
 * stable between renders, roughly uniform in -0.5..0.5.
 */
export function jitter(index: number): number {
  return ((index * 0.6180339887) % 1) - 0.5
}
