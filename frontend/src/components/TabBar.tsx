/** The app's tab strip: one place for the role/aria/active-class pattern. */
export function TabBar<T extends string>({
  tabs,
  current,
  onSelect,
}: {
  tabs: { id: T; label: string; tip?: string }[]
  current: T
  onSelect: (id: T) => void
}) {
  return (
    <div className="tab-bar" role="tablist">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          role="tab"
          aria-selected={current === tab.id}
          className={`tab${current === tab.id ? ' active' : ''}`}
          onClick={() => onSelect(tab.id)}
          title={tab.tip}
        >
          {tab.label}
        </button>
      ))}
    </div>
  )
}
