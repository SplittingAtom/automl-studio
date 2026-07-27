import type { InputSpecItem } from '../../api/schemas'

export function CategorySelect({
  item,
  value,
  onChange,
}: {
  item: InputSpecItem
  value: string
  onChange: (value: string) => void
}) {
  return (
    <div className="whatif-control">
      <div className="control-header">
        <label htmlFor={`select-${item.name}`}>{item.name}</label>
      </div>
      <select
        id={`select-${item.name}`}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {(item.options ?? []).map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  )
}
