import { studio } from '@/lib/studio'

/** First-run guide URL. */
const GETTING_STARTED_URL = 'https://inlinestudio.art/getting-started'

/**
 * A simple "Getting started" card for first-time users. Opens the getting-started guide
 * in the browser. Sits above the demo card on the home screen.
 */
export function GettingStartedCard(): React.JSX.Element {
  return (
    <button
      onClick={() => void studio().shell.openExternal(GETTING_STARTED_URL)}
      className="group flex w-full items-center gap-3 rounded-xl border border-accent/40 bg-accent/5 p-4 text-left transition-colors hover:border-accent"
    >
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-accent/15 text-accent">
        <CompassIcon />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-sm font-semibold text-zinc-100">
          New here? Start with the basics
        </span>
        <span className="block text-xs text-zinc-400">A quick guide to your first render.</span>
      </span>
      <ExternalLinkIcon />
    </button>
  )
}

function CompassIcon(): React.JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5">
      <circle cx="12" cy="12" r="9" />
      <path d="m15.5 8.5-2.1 4.9-4.9 2.1 2.1-4.9 4.9-2.1z" />
    </svg>
  )
}

function ExternalLinkIcon(): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      className="h-4 w-4 shrink-0 text-zinc-500 group-hover:text-accent"
    >
      <path d="M14 3h7v7" />
      <path d="M10 14 21 3" />
      <path d="M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5" />
    </svg>
  )
}
