/**
 * Small hover marker that explains a UI element in plain English. Uses the
 * native title tooltip so it works everywhere without a positioning library;
 * focusable so keyboard users can reach the aria-label too.
 */
export function InfoTip({ tip }: { tip: string }) {
  return (
    <span className="info-tip" title={tip} aria-label={tip} tabIndex={0} role="note">
      i
    </span>
  )
}
