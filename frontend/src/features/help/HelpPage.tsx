import { FEATURE_SECTIONS, HOWTO_SECTIONS, type HelpSection } from './helpContent'

/** Plain-English documentation: feature guide + task-oriented how-tos. */
export function HelpPage() {
  return (
    <div className="help-layout">
      <nav className="help-toc card" aria-label="Help contents">
        <h3>Features</h3>
        <ul>
          {FEATURE_SECTIONS.map((section) => (
            <li key={section.id}>
              <a href={`#${section.id}`}>{section.title}</a>
            </li>
          ))}
        </ul>
        <h3>How to…</h3>
        <ul>
          {HOWTO_SECTIONS.map((section) => (
            <li key={section.id}>
              <a href={`#${section.id}`}>{section.title}</a>
            </li>
          ))}
        </ul>
      </nav>
      <div className="help-body">
        <h1>Help &amp; documentation</h1>
        <p className="muted">
          Everything here is written for analysts — no machine-learning background
          needed. Hover the small <span className="info-tip">i</span> markers around the
          app for the same explanations in place.
        </p>
        <h2 className="help-group-title">Features</h2>
        {FEATURE_SECTIONS.map((section) => (
          <Section key={section.id} section={section} />
        ))}
        <h2 className="help-group-title">How to…</h2>
        {HOWTO_SECTIONS.map((section) => (
          <Section key={section.id} section={section} />
        ))}
      </div>
    </div>
  )
}

function Section({ section }: { section: HelpSection }) {
  return (
    <section id={section.id} className="card help-section">
      <h2>{section.title}</h2>
      {section.body}
    </section>
  )
}
