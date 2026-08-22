import { describe, expect, it } from 'vitest'

import { FEATURE_SECTIONS, HOWTO_SECTIONS } from '../features/help/helpContent'

describe('help content', () => {
  it('every section has a unique anchor id and a title', () => {
    const all = [...FEATURE_SECTIONS, ...HOWTO_SECTIONS]
    const ids = all.map((s) => s.id)
    expect(new Set(ids).size).toBe(ids.length)
    for (const section of all) {
      expect(section.id).toMatch(/^[a-z0-9-]+$/)
      expect(section.title.length).toBeGreaterThan(3)
      expect(section.body).toBeTruthy()
    }
  })

  it('covers the major features and tasks', () => {
    const titles = [...FEATURE_SECTIONS, ...HOWTO_SECTIONS]
      .map((s) => s.title.toLowerCase())
      .join(' ')
    for (const expected of ['explore', 'tuning', 'insights', 'forecast', 'first model', 'trustworthy']) {
      expect(titles).toContain(expected)
    }
  })
})
