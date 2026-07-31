import { describe, expect, it } from 'vitest'

import {
  PROMPT_SOURCE_HANDLE,
  PROMPT_TARGET_HANDLE,
  STARTER_RECIPES,
  recipeFor,
  starterLayout,
} from './starterRecipes'

/** The param keys every generation node in the repo exposes. A recipe key outside this set is a typo. */
const KNOWN_PARAM_KEYS = new Set([
  'width',
  'height',
  'steps',
  'guidance',
  'seed',
  'negative_prompt',
  'sampler',
  'scheduler',
  'strength',
  'variant',
  'model',
  'vae',
  'text_encoder',
  'controlnet',
  'control_strength',
])

const gen = STARTER_RECIPES.filter((r) => r.coreType !== null)

describe('starter recipes', () => {
  it('covers the four cards with unique keys', () => {
    expect(STARTER_RECIPES.map((r) => r.key)).toEqual(['zimage', 'flux2', 'krea2', 'training'])
    expect(new Set(STARTER_RECIPES.map((r) => r.key)).size).toBe(STARTER_RECIPES.length)
  })

  it('names real Core node types', () => {
    expect(gen.map((r) => r.coreType)).toEqual([
      'alibaba/z-image-turbo',
      'black-forest-labs/flux-2',
      'krea/krea-2-turbo',
    ])
  })

  it('only sets params the nodes actually have', () => {
    // Catches a `cfg` vs `guidance` slip at build time rather than as a silently ignored param.
    for (const recipe of gen)
      for (const key of Object.keys(recipe.params))
        expect(KNOWN_PARAM_KEYS.has(key), `${recipe.key} sets unknown param ${key}`).toBe(true)
  })

  it('leaves the file pickers alone', () => {
    // Core resolves these from disk; pinning a filename breaks on a differently named checkpoint.
    for (const recipe of gen)
      for (const key of ['model', 'vae', 'text_encoder'])
        expect(recipe.params).not.toHaveProperty(key)
  })

  it('keeps FLUX.2 on its from-the-checkpoint sentinels', () => {
    // steps 0 / guidance -1 let the runner use the detected variant's own schedule. A literal 4
    // would silently give anyone on the dev checkpoint the wrong number of steps.
    const flux = recipeFor('flux2')
    expect(flux?.params).toMatchObject({ steps: 0, guidance: -1 })
  })

  it('gives the distilled models their CFG-free schedule', () => {
    for (const key of ['zimage', 'krea2'] as const)
      expect(recipeFor(key)?.params).toMatchObject({ steps: 8, guidance: 0 })
  })

  it('ships a usable prompt with every generation card', () => {
    for (const recipe of gen) {
      expect(recipe.promptText.length).toBeGreaterThan(20)
      expect(recipe.promptText).not.toMatch(/[—–]/) // house style: no em or en dashes
    }
  })

  it('has no prompt or params on the training card, which builds no graph', () => {
    const training = recipeFor('training')
    expect(training?.coreType).toBeNull()
    expect(training?.promptText).toBe('')
    expect(training?.params).toEqual({})
  })

  it('lays the two nodes out without overlapping', () => {
    const { prompt, gen: genAt } = starterLayout({ x: 0, y: 0 })
    // Prompt nodes are 240 wide, so the model node has to start beyond that.
    expect(genAt.x).toBeGreaterThan(prompt.x + 240)
  })

  it('wires the prompt output into the model prompt input', () => {
    expect(PROMPT_SOURCE_HANDLE).toBe('out')
    expect(PROMPT_TARGET_HANDLE).toBe('prompt')
  })
})
