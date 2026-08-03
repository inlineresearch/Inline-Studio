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
  // H3 spells its own two differently: a video node counts seconds, and it inherited
  // `num_inference_steps` from the vendored blocks rather than the image nodes' `steps`.
  'duration',
  'num_inference_steps',
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
const withPrompt = STARTER_RECIPES.filter((r) => r.coreType !== null || r.falModelId)

describe('starter recipes', () => {
  it('covers the five cards with unique keys', () => {
    expect(STARTER_RECIPES.map((r) => r.key)).toEqual([
      'minimaxh3',
      'zimage',
      'flux2',
      'krea2',
      'training',
    ])
    expect(new Set(STARTER_RECIPES.map((r) => r.key)).size).toBe(STARTER_RECIPES.length)
  })

  it('never sets both a Core type and a fal model on one card', () => {
    for (const recipe of STARTER_RECIPES)
      expect(recipe.coreType != null && recipe.falModelId != null).toBe(false)
  })

  it('names real Core node types', () => {
    expect(gen.map((r) => r.coreType)).toEqual([
      'minimax/h3-text-to-video',
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

  it('starts MiniMax H3 on the local node, not the hosted one', () => {
    const h3 = recipeFor('minimaxh3')
    expect(h3?.coreType).toBe('minimax/h3-text-to-video')
    expect(h3?.falModelId).toBeUndefined()
  })

  it('asks H3 for a canvas that fits a real card', () => {
    // Not the node's 1344x768 default: the conditioner is resident at 19.5 GB, so the trained
    // canvas projects past 60 GB of VRAM at ten seconds. A first click must not open with an OOM.
    const h3 = recipeFor('minimaxh3')
    expect(h3?.params).toMatchObject({ width: 960, height: 544 })
    expect(Number(h3?.params.width) * Number(h3?.params.height)).toBeLessThan(1344 * 768)
  })

  it('labels each card with what it generates, which drives the colour coding', () => {
    expect(Object.fromEntries(STARTER_RECIPES.map((r) => [r.key, r.kind]))).toEqual({
      minimaxh3: 'video',
      zimage: 'image',
      flux2: 'image',
      krea2: 'image',
      training: 'training',
    })
  })

  it('tags the newest models, and says which kind of new H3 is', () => {
    const tagged = STARTER_RECIPES.filter((r) => r.tag).map((r) => r.key)
    expect(tagged).toEqual(['minimaxh3', 'flux2'])
    expect(recipeFor('minimaxh3')?.tag).toBe('Open weights')
    expect(recipeFor('flux2')?.tag).toBe('New')
  })

  it('ships a usable prompt with every generation card', () => {
    for (const recipe of withPrompt) {
      expect(recipe.promptText.length).toBeGreaterThan(20)
      expect(recipe.promptText).not.toMatch(/[—–]/) // house style: no em or en dashes
    }
  })

  it('has no prompt or params on the training card, which builds no graph', () => {
    const training = recipeFor('training')
    expect(training?.coreType).toBeNull()
    expect(training?.falModelId).toBeUndefined()
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
