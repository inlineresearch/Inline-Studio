import { useState } from 'react'
import { studio } from '@/lib/studio'

/**
 * Shown on the Generate tab when no ComfyUI is reachable. Two side-by-side cards
 * cover the two kinds of user: run ComfyUI locally on a GPU, or deploy it on a
 * cloud GPU (RunPod). Either way the user ends up pasting an address into the URL
 * field above this guide.
 */
/** YouTube id for the RunPod setup demo (used for both the thumbnail and the link). */
const DEMO_VIDEO_ID = 'JovhfHhxqdM'
const DEMO_VIDEO_URL = `https://youtu.be/${DEMO_VIDEO_ID}?si=lHQo9qzR_fCZwYCL`
/** An example of a RunPod public-proxy ComfyUI URL, to show users the shape. */
const EXAMPLE_RUNPOD_URL = 'https://2o3y58e14h8z63-8188.proxy.runpod.net'

export function ConnectionGuide(): React.JSX.Element {
  return (
    <div className="flex h-full flex-col items-center justify-center overflow-y-auto p-6">
      <div className="w-full max-w-3xl">
        <div className="mb-1 flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-zinc-600" />
          <h2 className="text-sm font-semibold text-zinc-200">ComfyUI is not connected</h2>
        </div>
        <p className="mb-5 text-xs text-zinc-500">
          Inline Studio renders every shot through your own ComfyUI. Pick how you want to run it,
          then paste its address into the URL field above and press Save.
        </p>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <LocalCard />
          <CloudCard />
        </div>

        <p className="mt-4 text-center text-xs text-zinc-500">
          …or any publicly hosted ComfyUI — just paste its URL above.
        </p>
      </div>
    </div>
  )
}

function LocalCard(): React.JSX.Element {
  return (
    <Card
      icon={<GpuIcon />}
      title="Local ComfyUI"
      subtitle="You have a GPU + ComfyUI on this machine"
    >
      <Step n={1}>
        Start ComfyUI as usual. Make sure to launch it with the{' '}
        <code className="rounded bg-black/40 px-1 text-[11px] text-accent">
          --enable-cors-header
        </code>{' '}
        flag — it lets Inline Studio talk to ComfyUI and is needed for full compatibility.
      </Step>
      <CopyBlock command="python main.py --enable-cors-header" />
      <Step n={2}>
        Copy the address ComfyUI prints (usually{' '}
        <code className="rounded bg-black/40 px-1 text-[11px] text-zinc-300">
          http://127.0.0.1:8188
        </code>
        ).
      </Step>
      <Step n={3}>Paste it into the URL field above and press Save.</Step>
    </Card>
  )
}

function CloudCard(): React.JSX.Element {
  return (
    <Card
      icon={<CloudIcon />}
      title="Cloud GPU (RunPod)"
      subtitle="No local GPU — deploy in the cloud"
    >
      <Step n={1}>Deploy ComfyUI on RunPod using an official template — pick one:</Step>
      <div className="flex flex-col gap-1.5">
        <TemplateLink
          label="ComfyUI"
          image="runpod/comfyui:latest"
          href="https://console.runpod.io/hub/template/comfyui?id=cw3nka7d08"
        />
        <TemplateLink
          label="ComfyUI · CUDA 13"
          image="runpod/comfyui:cuda13.0"
          href="https://console.runpod.io/hub/template/comfyui?id=cw3nka7d08"
        />
      </div>
      <Step n={2}>
        Once the pod is running, copy its <span className="text-zinc-300">public exposed</span>{' '}
        ComfyUI URL — it looks like:
      </Step>
      <code className="block truncate rounded-md border border-border bg-black/40 px-2 py-1.5 text-[11px] text-zinc-300">
        {EXAMPLE_RUNPOD_URL}
      </code>
      <Step n={3}>Paste that URL into the field above and press Save.</Step>
      <VideoThumb videoId={DEMO_VIDEO_ID} href={DEMO_VIDEO_URL} />
    </Card>
  )
}

/** Clickable YouTube thumbnail with a play overlay; opens the video in the browser. */
function VideoThumb({ videoId, href }: { videoId: string; href: string }): React.JSX.Element {
  return (
    <button
      onClick={() => void studio().shell.openExternal(href)}
      title="Watch the setup video on YouTube"
      className="group mt-0.5 overflow-hidden rounded-md border border-border hover:border-accent"
    >
      <span className="relative block">
        <img
          src={`https://img.youtube.com/vi/${videoId}/hqdefault.jpg`}
          alt="ComfyUI on RunPod — setup video"
          className="aspect-video w-full object-cover"
        />
        <span className="absolute inset-0 flex items-center justify-center bg-black/30 transition-colors group-hover:bg-black/15">
          <span className="flex h-9 w-9 items-center justify-center rounded-full bg-red-600 text-white shadow-lg">
            <PlayIcon />
          </span>
        </span>
        <span className="absolute bottom-1.5 left-1.5 rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-medium text-white">
          Watch the setup video
        </span>
      </span>
    </button>
  )
}

function Card({
  icon,
  title,
  subtitle,
  children,
}: {
  icon: React.ReactNode
  title: string
  subtitle: string
  children: React.ReactNode
}): React.JSX.Element {
  return (
    <div className="flex flex-col gap-2.5 rounded-lg border border-border bg-surface p-4">
      <div className="flex items-center gap-2.5">
        <span className="flex h-8 w-8 items-center justify-center rounded-md bg-black/30 text-accent">
          {icon}
        </span>
        <div>
          <h3 className="text-sm font-semibold text-zinc-100">{title}</h3>
          <p className="text-[11px] text-zinc-500">{subtitle}</p>
        </div>
      </div>
      <div className="flex flex-col gap-2.5 pt-1">{children}</div>
    </div>
  )
}

function Step({ n, children }: { n: number; children: React.ReactNode }): React.JSX.Element {
  return (
    <div className="flex gap-2 text-xs leading-relaxed text-zinc-400">
      <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-accent/20 text-[10px] font-semibold text-accent">
        {n}
      </span>
      <p className="min-w-0">{children}</p>
    </div>
  )
}

function CopyBlock({ command }: { command: string }): React.JSX.Element {
  const [copied, setCopied] = useState(false)
  const copy = (): void => {
    void navigator.clipboard.writeText(command).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }
  return (
    <div className="flex items-center gap-2 rounded-md border border-border bg-black/40 px-2 py-1.5">
      <code className="min-w-0 flex-1 truncate text-[11px] text-zinc-200">{command}</code>
      <button
        onClick={copy}
        title="Copy command"
        className="shrink-0 rounded px-1.5 py-0.5 text-[10px] text-zinc-400 hover:bg-surface hover:text-white"
      >
        {copied ? 'Copied' : 'Copy'}
      </button>
    </div>
  )
}

function TemplateLink({
  label,
  image,
  href,
}: {
  label: string
  image: string
  href: string
}): React.JSX.Element {
  return (
    <button
      onClick={() => void studio().shell.openExternal(href)}
      className="group flex items-center justify-between gap-2 rounded-md border border-border bg-black/20 px-2.5 py-1.5 text-left hover:border-accent hover:bg-surface"
    >
      <span className="min-w-0">
        <span className="block text-xs font-medium text-zinc-200">{label}</span>
        <code className="block truncate text-[10px] text-zinc-500">{image}</code>
      </span>
      <ExternalIcon />
    </button>
  )
}

function GpuIcon(): React.JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-4 w-4">
      <rect x="2" y="6" width="20" height="12" rx="2" />
      <rect x="6" y="10" width="6" height="4" rx="1" />
      <line x1="16" y1="10" x2="16" y2="14" />
      <line x1="19" y1="10" x2="19" y2="14" />
    </svg>
  )
}

function CloudIcon(): React.JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className="h-4 w-4">
      <path d="M19.35 10.04A7.49 7.49 0 0 0 12 4C9.11 4 6.6 5.64 5.35 8.04A5.994 5.994 0 0 0 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96z" />
    </svg>
  )
}

function PlayIcon(): React.JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className="h-3.5 w-3.5">
      <path d="M8 5v14l11-7z" />
    </svg>
  )
}

function ExternalIcon(): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      className="h-3.5 w-3.5 shrink-0 text-zinc-500 group-hover:text-accent"
    >
      <path d="M14 3h7v7" />
      <path d="M10 14 21 3" />
      <path d="M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5" />
    </svg>
  )
}
