/**
 * Plain-English preview of what training will actually do, computed from the
 * chosen settings. Constants mirror the backend trainer exactly:
 * TEST_FRACTION=0.2, EARLY_STOPPING_MIN_ROWS=200, TUNING_TRIALS=12,
 * CV_FOLDS=5, max_training_rows=100k.
 */

const TEST_FRACTION = 0.2
const EARLY_STOPPING_MIN_ROWS = 200
const TUNING_TRIALS = 12
const CV_FOLDS = 5
const MAX_TRAINING_ROWS = 100_000

export interface PreviewInput {
  rowCount: number
  thorough: boolean
  timeColumn: string | null
  horizon: number
  tunedSettings: number
}

export function experimentPreview(input: PreviewInput): string[] {
  const usedRows = Math.min(input.rowCount, MAX_TRAINING_ROWS)
  const testRows = Math.round(usedRows * TEST_FRACTION)
  const lines: string[] = []

  if (input.rowCount > MAX_TRAINING_ROWS) {
    lines.push(
      input.timeColumn
        ? `Trains on the most recent ${MAX_TRAINING_ROWS.toLocaleString()} rows for speed.`
        : `Trains on a random ${MAX_TRAINING_ROWS.toLocaleString()}-row sample for speed.`,
    )
  }

  if (input.timeColumn) {
    lines.push(
      `Tests on the most recent ${testRows.toLocaleString()} rows (ordered by ${input.timeColumn}) — the model never sees the future while learning.`,
    )
    if (input.horizon > 0) {
      lines.push(
        `Leaves a ${input.horizon}-row gap before the test period so a ${input.horizon}-step-ahead target can't leak into it.`,
      )
    }
  } else {
    lines.push(
      `Holds back a random ${testRows.toLocaleString()} rows (20%) — scores come only from data the model never sees.`,
    )
  }

  if (usedRows - testRows >= EARLY_STOPPING_MIN_ROWS) {
    lines.push('Stops training automatically once improvements stall.')
  }

  lines.push(
    input.thorough
      ? `Tests ${TUNING_TRIALS} model variations and keeps the best one.`
      : 'Trains one model with sensible defaults.',
  )

  if (input.tunedSettings > 0) {
    lines.push(
      `Pins ${input.tunedSettings} advanced setting${input.tunedSettings === 1 ? '' : 's'} to your values.`,
    )
  }

  lines.push(
    `Double-checks stability with ${CV_FOLDS} re-tests on different slices (cross-validation).`,
  )
  return lines
}
