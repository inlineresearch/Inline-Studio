/** The Add-menu node kinds, and how each one reaches a renderer. */

export const ADD_NODE_KINDS = [
  'load',
  'layer',
  'preview',
  'director',
  'trim',
  'prompt',
  'controlSpace',
  'train/dataset',
  'train/caption',
  'train/lora',
  'train/loss',
  'resource',
] as const

/** The node kinds the Add menu can create (Text has its own toolbar tool, so it's not here). */
export type AddNodeKind = (typeof ADD_NODE_KINDS)[number]

/** Kinds `nodeFor` maps with a branch of their own. */
export const EXPLICIT_RENDER_KINDS: ReadonlySet<string> = new Set([
  'load',
  'layer',
  'preview',
  'director',
  'trim',
  'prompt',
  'controlSpace',
])

/**
 * Kinds whose node reads the board by item id rather than an asset. Named for the rule, not for the
 * Training menu: `resource` sat outside a set called TRAINING_TYPES and so fell through to the
 * asset branch, rendering an assetless blank that read as the node never being added at all.
 */
export const BY_ID_TYPES: ReadonlySet<string> = new Set([
  'train/dataset',
  'train/caption',
  'train/lora',
  'train/loss',
  'resource',
])
