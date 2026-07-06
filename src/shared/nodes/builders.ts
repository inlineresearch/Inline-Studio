/**
 * Shared, pure helpers for declaring fal generation models (NodeDefs) concisely. Every model reuses
 * these instead of re-implementing endpoint/output/price/param boilerplate — so adding a model is a
 * short, declarative file. Kept shell-agnostic (imported by both renderer and main).
 */
import {
  extFromContentTypeOrName,
  type FalOutputRef,
  type ParamField,
  type PriceEstimate,
} from './types'

/** A model whose fal endpoint never varies with its inputs (the common case). */
export function constantEndpoint(id: string): () => string {
  return () => id
}

interface ImageResult {
  url?: string
  content_type?: string
  file_name?: string
}
interface VideoResult {
  url?: string
  content_type?: string
  file_name?: string
}

/** Parse a `{ images: [{ url, content_type, file_name }] }` response into output refs. */
export function parseImageArray(response: unknown, defaultExt = '.png'): FalOutputRef[] {
  const images = (response as { images?: ImageResult[] } | null)?.images ?? []
  return images
    .filter(
      (img): img is ImageResult & { url: string } =>
        typeof img?.url === 'string' && img.url.length > 0,
    )
    .map((img) => ({
      url: img.url,
      ext: extFromContentTypeOrName(img.content_type, img.file_name, defaultExt),
      kind: 'image' as const,
    }))
}

/** Parse a `{ video: { url, content_type, file_name } }` response into a single output ref. */
export function parseSingleVideo(response: unknown): FalOutputRef[] {
  const video = (response as { video?: VideoResult } | null)?.video
  if (!video || typeof video.url !== 'string' || video.url.length === 0) return []
  return [
    {
      url: video.url,
      ext: extFromContentTypeOrName(video.content_type, video.file_name, '.mp4'),
      kind: 'video' as const,
    },
  ]
}

/** A rough (always-approximate) USD price estimate. */
export function approxPrice(amount: number): PriceEstimate {
  return { amount, approx: true }
}

/** A select param from a plain list of values (value === label). Advanced by default. */
export function selectParam(
  key: string,
  label: string,
  values: readonly string[],
  defaultValue: string,
  opts: { advanced?: boolean } = {},
): ParamField {
  return {
    key,
    label,
    widget: 'select',
    options: values.map((v) => ({ value: v, label: v })),
    default: defaultValue,
    advanced: opts.advanced ?? true,
  }
}

/** A number param. Advanced by default. */
export function numberParam(
  key: string,
  label: string,
  defaultValue: number,
  opts: { min?: number; max?: number; step?: number; advanced?: boolean } = {},
): ParamField {
  return {
    key,
    label,
    widget: 'number',
    default: defaultValue,
    min: opts.min,
    max: opts.max,
    step: opts.step,
    advanced: opts.advanced ?? true,
  }
}

/** A boolean param. Advanced by default. */
export function booleanParam(
  key: string,
  label: string,
  defaultValue: boolean,
  opts: { advanced?: boolean } = {},
): ParamField {
  return { key, label, widget: 'boolean', default: defaultValue, advanced: opts.advanced ?? true }
}

/** The common "seed" param — `-1` means a random seed (omitted from the request). */
export function seedParam(): ParamField {
  return {
    key: 'seed',
    label: 'Seed (-1 = random)',
    widget: 'number',
    default: -1,
    step: 1,
    advanced: true,
  }
}

/** Add `seed` to a request body only when it's a finite, non-negative value. */
export function putSeed(body: Record<string, unknown>, params: Record<string, unknown>): void {
  const seed = Number(params.seed ?? -1)
  if (Number.isFinite(seed) && seed >= 0) body.seed = seed
}
