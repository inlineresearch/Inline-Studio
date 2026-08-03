/**
 * The starter graphs behind the getting-started cards: one prompt node wired into one model node,
 * with params and a prompt already filled in so the first click ends in a render.
 *
 * Pure data, so the definitions can be tested against the live node descriptors without a canvas.
 */
import type { StarterKey } from './vramAdvice'

export interface StarterRecipe {
  key: StarterKey
  /** Inline Core node type, or null for a card that builds no Core node (API nodes, training). */
  coreType: string | null
  /** fal model id for a card that starts on a hosted model instead. Mutually exclusive with above. */
  falModelId?: string
  title: string
  /** Short chip beside the title, e.g. `New`. Absent on the settled cards. */
  tag?: string
  /** What this card generates. Drives the card's colour coding. */
  kind: 'image' | 'video' | 'training'
  blurb: string
  params: Record<string, unknown>
  promptText: string
}

/**
 * `model`, `vae` and `text_encoder` are deliberately absent from every recipe. Core resolves those
 * from what is on disk and now serves the resolved filename as the param default, so writing one
 * here would pin a filename that breaks the moment the user has a differently named checkpoint.
 */
export const STARTER_RECIPES: readonly StarterRecipe[] = [
  {
    key: 'minimaxh3',
    coreType: 'minimax/h3-text-to-video',
    title: 'MiniMax H3',
    tag: 'Open weights',
    kind: 'video',
    blurb: 'Video and its 32 kHz soundtrack in one pass, on your own GPU. A 144 GB download.',
    // 960x544 rather than the node's 1344x768 default, and it is not a taste call: the conditioner
    // is resident at 19.5 GB, so the trained canvas projects to about 65 GB of VRAM at ten seconds
    // and does not fit a 45 GB card. 20 steps matches the reference templates.
    params: { width: 960, height: 544, duration: 5.17, num_inference_steps: 20, seed: -1 },
    // Video prompts read as prose, not the comma-separated fragments the image cards use: the scene
    // first, then the camera move and the light. That is how MiniMax's own examples are written.
    // The soundtrack is described too, because H3 denoises it alongside the picture rather than
    // dubbing it afterwards, and a prompt that says nothing about sound gets sound anyway.
    promptText:
      'A photoreal city street in the late afternoon, where the buildings and pedestrians are live action but every tree and parked car is a blocky voxel object built from visible cubes with low-resolution textures. The camera tracks slowly along the pavement, warm low sun, real shadows falling across the filmed and the voxel elements alike. Audio: street ambience, distant traffic, footsteps on pavement.',
  },
  {
    key: 'zimage',
    coreType: 'alibaba/z-image-turbo',
    title: 'Z-Image Turbo',
    kind: 'image',
    blurb: 'Eight steps, no guidance. The quickest way to a first image.',
    params: { width: 1024, height: 1024, steps: 8, guidance: 0, seed: -1 },
    promptText:
      'A tall white lighthouse on a cliff above a stormy sea at sunset, beam lit, dramatic clouds, cinematic',
  },
  {
    key: 'flux2',
    coreType: 'black-forest-labs/flux-2',
    title: 'FLUX.2',
    tag: 'New',
    kind: 'image',
    blurb: 'Prose prompts and multi-reference editing. Klein 4B is the light build.',
    // steps 0 and guidance -1 mean "from the checkpoint": the runner substitutes the detected
    // variant's own schedule (klein wants 4 steps at guidance 1.0, dev wants 28 at 4.0). Writing a
    // literal here would silently give dev users the wrong schedule.
    params: { width: 1024, height: 1024, steps: 0, guidance: -1, seed: -1 },
    promptText:
      'A retro-futuristic diner on Mars, neon signage, chrome and dust, wide establishing shot, film still',
  },
  {
    key: 'krea2',
    coreType: 'krea/krea-2-turbo',
    title: 'Krea 2 Turbo',
    kind: 'image',
    blurb: 'Photographic look, distilled to eight steps. The heaviest of the three.',
    params: { width: 1024, height: 1024, steps: 8, guidance: 0, seed: -1 },
    promptText:
      'Close-up portrait in soft window light, natural skin texture, 85mm lens, shallow depth of field',
  },
  {
    key: 'training',
    coreType: null,
    title: 'Train a LoRA',
    kind: 'training',
    blurb: 'Teach a style or subject from your own images, then generate with it.',
    params: {},
    promptText: '',
  },
]

export function recipeFor(key: StarterKey): StarterRecipe | undefined {
  return STARTER_RECIPES.find((r) => r.key === key)
}

export interface Point {
  x: number
  y: number
}

/**
 * Where the two nodes land, relative to the viewport centre. Prompt nodes default to 240x120 and
 * core nodes to 200x120, so this clears both with room for the connector.
 */
export function starterLayout(centre: Point): { prompt: Point; gen: Point } {
  return {
    prompt: { x: centre.x - 380, y: centre.y - 20 },
    gen: { x: centre.x + 20, y: centre.y - 60 },
  }
}

/** Handles the connector uses: a prompt node's only output into the model's prompt input. */
export const PROMPT_SOURCE_HANDLE = 'out'
export const PROMPT_TARGET_HANDLE = 'prompt'
