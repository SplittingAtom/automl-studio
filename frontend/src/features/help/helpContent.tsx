import type { ReactNode } from 'react'

export interface HelpSection {
  id: string
  title: string
  body: ReactNode
}

/** Feature guide: what each part of the app does and how to read it. */
export const FEATURE_SECTIONS: HelpSection[] = [
  {
    id: 'loading-data',
    title: 'Loading your data',
    body: (
      <>
        <p>
          Upload a CSV or Excel file, or start from one of the bundled sample datasets.
          Every column is automatically classified as a <strong>Number</strong>,{' '}
          <strong>Category</strong>, <strong>Date</strong>, <strong>ID</strong>, or{' '}
          <strong>Empty</strong> — the type decides how the column is used: numbers get
          sliders and histograms, categories get dropdowns and bar charts, dates become
          year/month/weekday signals, and ID-like or empty columns are left out of models
          automatically.
        </p>
        <p>
          Anything worth knowing (a column that looks like an ID, heavy missing values, a
          column of dates) appears as a yellow notice. Notices never block you — they
          explain what the app decided and why.
        </p>
      </>
    ),
  },
  {
    id: 'explore-data',
    title: 'Explore data (profiling)',
    body: (
      <>
        <p>
          When you open a dataset you land on the <strong>Explore data</strong> tab: an
          overview of rows, columns, empty cells, and exact duplicate rows, then one card
          per column showing its distribution — a histogram for numbers, top categories
          for categories, a timeline for dates — plus min / median / mean / max, spread,
          missing share, and how many values sit outside the typical range (outliers).
        </p>
        <p>
          <strong>Worth a look</strong> lists only the statistically notable things:
          columns stretched by a few huge values, categories that are almost always the
          same, pairs of columns that carry the same information, and duplicate rows.
          If it's quiet, your data is clean.
        </p>
        <p>
          <strong>Add a calculated column</strong> lets you combine columns with a
          formula, like <code>fare / (sibsp + parch + 1)</code>. This creates a new copy
          of the dataset — the original is never changed — and the new column works
          everywhere: charts, training, even as the thing to predict.
        </p>
      </>
    ),
  },
  {
    id: 'choosing-target',
    title: 'Dataset analysis & choosing what to predict',
    body: (
      <>
        <p>
          The <strong>Set up model</strong> tab starts with an automatic analysis: the
          app quickly test-predicts every eligible column from the others and ranks them
          by how learnable they are, with a plain-English reason and the strongest
          predictor columns for each. A high score means the other columns genuinely
          carry signal about that one.
        </p>
        <p>
          Columns that are <em>too</em> predictable (near-perfect from one other column)
          are marked as probably calculated from other columns — predicting them is
          usually circular, so they're never recommended.
        </p>
      </>
    ),
  },
  {
    id: 'setup',
    title: 'Setting up a model',
    body: (
      <>
        <p>
          Pick the column to predict; the app detects whether that means predicting a{' '}
          <strong>category</strong> or a <strong>number</strong> (you can override it).
          Untick any column you want left out — the model reports afterwards whether a
          column was pulling its weight, so when unsure, leave it in and compare runs.
        </p>
        <p>
          <strong>Time-ordered data?</strong> If your rows form a timeline (daily sales,
          sensor readings), choose the date column that orders them. The model is then
          tested only on the most recent rows — it never peeks at the future — and gains
          recent-history columns built from past values of the target. If your target
          looks N rows ahead (say, a 10-day return), set the{' '}
          <strong>prediction horizon</strong> to N so a gap that size protects the test
          period.
        </p>
        <p>
          <strong>Try harder</strong> tests a dozen model variations and keeps the best.{' '}
          <strong>What will happen</strong> tells you the exact plan — split, early
          stopping, re-tests — before you commit. <strong>Build model</strong> trains
          one model; <strong>Auto-compare 3 approaches</strong> trains standard,
          thorough, and simpler-steadier variants and ranks them for you.
        </p>
      </>
    ),
  },
  {
    id: 'tuning',
    title: 'Advanced tuning & direction rules',
    body: (
      <>
        <p>
          The collapsed <strong>Advanced tuning</strong> panel is for fine-tuning after
          you have a baseline. Each knob has a plain-English name, its technical name in
          small print, a safe range, and a one-line hint. Settings you pin stay fixed
          even during a "try harder" search. Every tuned run is a <em>new</em> model —
          your baseline is never touched.
        </p>
        <p>
          <strong>Direction rules</strong> force the prediction to only rise (or only
          fall) as a column grows — useful when you know the real-world relationship,
          e.g. "price only goes up with square footage". They make the model easier to
          trust, occasionally at a small cost in score.
        </p>
        <p>
          The easiest way in: open a finished model and press <strong>Fine-tune</strong>.
          Everything is pre-filled from that run, and the new model's page shows exactly
          how it compares to the one you started from.
        </p>
      </>
    ),
  },
  {
    id: 'results',
    title: 'Reading the results page',
    body: (
      <>
        <p>
          The headline cards show the model's quality on rows it never saw during
          training, with context lines underneath: what naive guessing would score, what
          a basic statistical model scores, and a consistency range from re-testing on
          five different slices of the data. If the fancy model barely beats the basic
          one, the app says so — that's honesty, not failure.
        </p>
        <p>
          For yes/no predictions, <strong>Tune the decision cut-off</strong> trades
          catching more true cases against more false alarms, and{' '}
          <strong>Can you trust the percentages?</strong> checks whether "70% sure"
          really happens 70% of the time.
        </p>
        <p>
          <strong>What drives the predictions?</strong> shows each column's share of the
          model's decisions. <strong>How does … affect the prediction?</strong> sweeps
          one column across its range while everything else stays at your current
          what-if values. <strong>Explore what-if scenarios</strong> is the live
          panel: set inputs, watch the prediction, and see which values pushed it up or
          down. <strong>Proof on rows the model never saw</strong> is the raw held-out
          table — download it as CSV to audit any prediction.
        </p>
        <p>
          The app may also suggest one-click follow-ups: retraining without a column that
          contributed nothing (or one that looks suspiciously like the answer), and{' '}
          <strong>Column ideas</strong> — simple formulas built from your strongest
          columns that a quick test found genuinely useful.
        </p>
      </>
    ),
  },
  {
    id: 'column-insights',
    title: 'Column insights (explainability)',
    body: (
      <>
        <p>
          <strong>How column values push predictions</strong>: each dot is one held-out
          row; its position shows how hard that column pushed the prediction, and its
          color shows the column's value (light = low, dark = high, gray = a category or
          missing value). A column whose dark dots sit right and light dots sit left has
          a clean "higher value → higher prediction" story.
        </p>
        <p>
          <strong>Where the model struggles</strong> flags groups of rows (by category)
          that score notably worse than average — check it before relying on the model
          for everyone. <strong>How the model decides — simplified</strong> is a small
          flowchart approximation with an honest note about how closely it matches the
          real model. <strong>How columns relate</strong> is the relationship heatmap:
          darker = more strongly related; the last row/column is the prediction itself,
          and numeric pairs keep their sign (negative = one falls as the other rises).
        </p>
      </>
    ),
  },
  {
    id: 'forecasting',
    title: 'Time-series prediction & forecasting',
    body: (
      <>
        <p>
          Models trained with a time column (predicting a number) get a{' '}
          <strong>Time-series prediction</strong> tab: actual vs predicted over the
          held-out recent window, plus <strong>Forecast ahead</strong> — the model rolls
          forward day by day, feeding each prediction into the next step, with a shaded
          band showing the plausible range.
        </p>
        <p>
          Expect forecasts to flatten toward recent levels the further out you go —
          tree-based models don't extrapolate trends. And expect time-aware scores to be
          lower than a regular model's on the same data: they're the honest number,
          because the model was never allowed to peek at the future.
        </p>
      </>
    ),
  },
  {
    id: 'comparing',
    title: 'Comparing runs',
    body: (
      <>
        <p>
          Every training run is kept. <strong>Compare models</strong> lists them ranked —
          runs still training first, then best score — with the top comparable run tagged{' '}
          <strong>Best</strong>. Runs are named by how they were made (Standard, Thorough
          search, Fine-tuned, or an auto-compare label), and a fine-tuned model's page
          always shows its delta against the run it started from.
        </p>
      </>
    ),
  },
  {
    id: 'sharing',
    title: 'Sharing & using the model elsewhere',
    body: (
      <>
        <p>
          <strong>Download report</strong> produces a one-page HTML file — scores in
          plain English, what drives predictions, caveats, and the settings used — ready
          to email to a colleague. <strong>Download model</strong> produces a
          self-contained scoring kit: the trained model plus a <code>predict.py</code>{' '}
          that runs anywhere with Python, no this-app-required. The validation table's
          CSV download gives auditors the raw held-out evidence.
        </p>
      </>
    ),
  },
]

/** Task-oriented walkthroughs. */
export const HOWTO_SECTIONS: HelpSection[] = [
  {
    id: 'howto-first-model',
    title: 'Train your first model',
    body: (
      <ol>
        <li>On the home page, upload a CSV/Excel file or pick a sample dataset.</li>
        <li>Skim the <strong>Explore data</strong> tab — check "Worth a look" for surprises.</li>
        <li>Open <strong>Set up model</strong> and pick a recommended column to predict.</li>
        <li>Press <strong>Build model</strong>. Training usually takes seconds.</li>
        <li>
          Read the headline cards <em>and</em> the context lines under them — a model is
          only good if it clearly beats the simple alternatives shown there.
        </li>
      </ol>
    ),
  },
  {
    id: 'howto-improve',
    title: 'Make a model better',
    body: (
      <ol>
        <li>
          Start from the results page: apply any "retrain without it" suggestion, and add
          any <strong>Column ideas</strong> that look sensible.
        </li>
        <li>
          Use <strong>Auto-compare 3 approaches</strong> to see whether thorough search
          or a simpler model wins on your data.
        </li>
        <li>
          Then press <strong>Fine-tune</strong> on the best run and adjust one knob at a
          time — each run shows its delta vs the baseline, so you always know whether a
          change helped.
        </li>
        <li>
          More signal beats more tuning: a good calculated column (a ratio, a
          per-something value) often helps more than any knob.
        </li>
      </ol>
    ),
  },
  {
    id: 'howto-explain',
    title: 'Understand why the model predicts what it does',
    body: (
      <ol>
        <li>
          For one prediction: set the inputs in <strong>Explore what-if scenarios</strong>{' '}
          and read the push-bars under the prediction.
        </li>
        <li>
          For one column: use <strong>How does … affect the prediction?</strong> to sweep
          it across its range.
        </li>
        <li>
          For the whole model: open <strong>Column insights</strong> — the dot chart for
          how values push predictions, the flowchart for the big picture, and the heatmap
          for how columns relate.
        </li>
      </ol>
    ),
  },
  {
    id: 'howto-forecast',
    title: 'Forecast future values',
    body: (
      <ol>
        <li>Your data needs a date column and a number to predict.</li>
        <li>
          In <strong>Set up model</strong>, choose the date column under{' '}
          <strong>Time-ordered data?</strong> (set a horizon if your target looks N rows
          ahead).
        </li>
        <li>
          After training, open the <strong>Time-series prediction</strong> tab, check how
          well the model tracked the held-out weeks, then set how many steps to forecast
          ahead.
        </li>
      </ol>
    ),
  },
  {
    id: 'howto-trust',
    title: 'Check a model is trustworthy',
    body: (
      <ol>
        <li>
          <strong>Consistency line</strong>: the score range from five re-tests — a wide
          range means the score is shaky.
        </li>
        <li>
          <strong>Proof table</strong>: spot-check real held-out rows, or download the CSV.
        </li>
        <li>
          <strong>Can you trust the percentages?</strong>: whether stated confidence
          matches reality.
        </li>
        <li>
          <strong>Where the model struggles</strong>: groups served worse than average.
        </li>
        <li>
          A "suspiciously perfect" notice means a column probably contains the answer —
          retrain without it before believing the score.
        </li>
      </ol>
    ),
  },
  {
    id: 'howto-export',
    title: 'Use the model outside this app',
    body: (
      <ol>
        <li>
          Press <strong>Download model</strong> on the results page to get a zip with the
          model, its input contract, and a standalone <code>predict.py</code>.
        </li>
        <li>
          On any machine with Python: unzip, install the two listed packages, then run{' '}
          <code>python predict.py your_new_rows.csv</code> — it writes predictions next
          to your data.
        </li>
        <li>
          Send <strong>Download report</strong> to stakeholders instead — it's the
          plain-English summary of what the model is and how good it is.
        </li>
      </ol>
    ),
  },
]
