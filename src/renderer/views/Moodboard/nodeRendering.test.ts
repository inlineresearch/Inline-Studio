/** Every node the add menu offers must actually render; a missing mapping looks like a dead menu. */
import { describe, expect, it } from 'vitest'
import { ADD_NODE_KINDS, BY_ID_TYPES, EXPLICIT_RENDER_KINDS } from './nodeKinds'

describe('add-menu kinds all reach a renderer', () => {
  it('classifies every kind the menu can create', () => {
    const orphans = ADD_NODE_KINDS.filter(
      (kind) => !EXPLICIT_RENDER_KINDS.has(kind) && !BY_ID_TYPES.has(kind),
    )
    // An unclassified kind falls through nodeFor to the asset branch and renders an assetless
    // blank, which on the canvas is indistinguishable from the add having done nothing.
    expect(orphans).toEqual([])
  })

  it('includes Resources, which shipped missing and read as an add that did nothing', () => {
    expect(BY_ID_TYPES.has('resource')).toBe(true)
  })
})
