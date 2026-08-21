import { studio } from '@/lib/studio'
import { MovieIcon } from '../../components/icons'

/** The MiniMax H3 LoRA training guide on the site. */
const H3_TRAINING_URL = 'https://inlinestudio.art/lora-training/minimax-h3'

/**
 * A home-screen card for the newest thing the app does: train a MiniMax H3 LoRA on your own clips,
 * on your own GPU. Sits under the "New here?" card and opens the training guide.
 */
export function H3TrainingCard(): React.JSX.Element {
  return (
    <button
      onClick={() => void studio().shell.openExternal(H3_TRAINING_URL)}
      className="group flex w-full items-center gap-3 rounded-xl border border-emerald-500/40 bg-emerald-500/5 p-4 text-left transition-colors hover:border-emerald-500"
    >
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-emerald-500/15 text-emerald-400">
        <MovieIcon className="h-5 w-5" />
      </span>
      <span className="min-w-0 flex-1">
        <span className="flex items-center gap-2">
          <span className="text-sm font-semibold text-zinc-100">MiniMax H3 LoRA training</span>
          <span className="shrink-0 rounded-full bg-emerald-500/20 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-emerald-300">
            New
          </span>
        </span>
        <span className="block text-xs text-zinc-400">
          Train a LoRA on your own clips, on your own GPU. Your footage never leaves the machine.
        </span>
      </span>
      <ExternalLinkIcon />
    </button>
  )
}

function ExternalLinkIcon(): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      className="h-4 w-4 shrink-0 text-zinc-500 group-hover:text-emerald-400"
    >
      <path d="M14 3h7v7" />
      <path d="M10 14 21 3" />
      <path d="M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5" />
    </svg>
  )
}
