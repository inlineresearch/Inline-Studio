/** Where users create a fal.ai API key (also referenced by the Settings key field). */
const FAL_KEYS_URL = 'https://fal.ai/dashboard/keys'

/**
 * A home-screen card pitching the fal (Generate) engine: run closed-source models straight from the
 * canvas with no ComfyUI install — you just bring your own fal.ai key. Sits under the "New here?"
 * card and opens fal.ai's keys dashboard so new users can grab a key (it stays on their device).
 */
export function FalKeyCard(): React.JSX.Element {
  return (
    <button
      onClick={() => void window.inlineStudio.shell.openExternal(FAL_KEYS_URL)}
      className="group flex w-full items-center gap-3 rounded-xl border border-emerald-500/40 bg-emerald-500/5 p-4 text-left transition-colors hover:border-emerald-500"
    >
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-emerald-500/15 text-emerald-400">
        <FalIcon />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-sm font-semibold text-zinc-100">
          Closed models, no ComfyUI setup
        </span>
        <span className="block text-xs text-zinc-400">
          Run GPT Image, Seedance &amp; more right on the canvas — nothing to install. Bring your
          own fal.ai API key; it stays on your device.
        </span>
      </span>
      <ExternalLinkIcon />
    </button>
  )
}

/** The fal mark: a rounded four-pointed star with a hollow centre. */
function FalIcon(): React.JSX.Element {
  return (
    <svg viewBox="0 0 640 640" className="h-5 w-5" aria-hidden="true">
      <path
        fill="currentColor"
        fillRule="evenodd"
        clipRule="evenodd"
        d="M204 110A94 94 0 0 1 298 16L342 16A94 94 0 0 1 436 110L436 110A94 94 0 0 0 530 204L530 204A94 94 0 0 1 624 298L624 342A94 94 0 0 1 530 436L530 436A94 94 0 0 0 436 530L436 530A94 94 0 0 1 342 624L298 624A94 94 0 0 1 204 530L204 530A94 94 0 0 0 110 436L110 436A94 94 0 0 1 16 342L16 298A94 94 0 0 1 110 204L110 204A94 94 0 0 0 204 110ZM320 170A150 150 0 1 0 320 470A150 150 0 1 0 320 170Z"
      />
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
      className="h-4 w-4 shrink-0 text-zinc-500 group-hover:text-emerald-400"
    >
      <path d="M14 3h7v7" />
      <path d="M10 14 21 3" />
      <path d="M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5" />
    </svg>
  )
}
