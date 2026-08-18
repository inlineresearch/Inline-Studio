import type { RegistryModel } from '@shared/types'

/** Where a model comes from, as a page someone can open: the file on the hub, or its own link. */
export function sourceUrl(model: RegistryModel): string {
  if (model.kind === 'url') return model.url
  if (!model.repo) return ''
  const view = model.kind === 'hf_folder' ? 'tree' : 'blob'
  return `https://huggingface.co/${model.repo}/${view}/main/${model.path}`
}
