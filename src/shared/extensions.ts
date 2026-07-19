/**
 * Community extensions: the wire shapes for the `extensions:*` channels.
 *
 * Mirrors the Python dataclasses in `inline_core/extensions/`. An extension extension is one repo at one
 * pinned commit, holding module-extensions that can be toggled independently.
 */

/** How severely a scan finding gates the install. */
export type FindingSeverity = 'critical' | 'high' | 'medium' | 'low'

export interface ScanFinding {
  severity: FindingSeverity
  /** Stable rule id, e.g. `subprocess`. Consent is recorded per rule. */
  rule: string
  message: string
  file: string
  line: number
}

export interface ScanReport {
  /** A CRITICAL finding: the install is refused and cannot be overridden. */
  blocked: boolean
  /** HIGH/MEDIUM findings the user must explicitly accept. */
  needsConsent: boolean
  consentRules: string[]
  findings: ScanFinding[]
}

/** A dependency the extension wants that the host already owns at another version. */
export interface DependencyConflict {
  name: string
  hostVersion: string
  wanted: string
  /** Part of the shared ML runtime (torch, diffusers, …) - never replaceable. */
  protected: boolean
  message: string
}

/** One node an extension provides, and whether the user has it switched on. */
export interface ExtensionNodeInfo {
  type: string
  title: string
  enabled: boolean
}

export interface ExtensionInfo {
  extensionId: string
  name: string
  description: string
  version: string
  license: string
  /** The URL it was installed from, for the repository link and the update check. */
  repo: string
  /** The tag or branch asked for, and the commit it resolved to. */
  ref: string
  sha: string
  /** The installed version directory, e.g. `1.4.0+9f2c1ab`. */
  installed: string
  enabled: boolean
  /** Every version kept on disk, newest first - the rollback menu. */
  versions: string[]
  homepage: string
  nodes: ExtensionNodeInfo[]
}

/** An external tool the installer needs (`git`, `uv`). */
export interface ExtensionTool {
  name: string
  available: boolean
  version: string | null
  /** Platform-specific install command, shown when `available` is false. */
  hint: string
}

export interface ExtensionsStatus {
  canInstall: boolean
  tools: ExtensionTool[]
  extensions: ExtensionInfo[]
}

/** One entry in the published registry index. */
export interface RegistryEntry {
  id: string
  name: string
  description: string
  repo: string
  homepage?: string
  author?: string
  tags?: string[]
  /** Rarely set: freezes a listing to one tag. Normally absent, and the newest tag wins. */
  pin?: string
}

/** Whether an extension's ref has moved upstream since it was installed. */
export interface UpdateStatus {
  extensionId: string
  ref: string
  installedSha: string
  remoteSha: string | null
  /** The newest release tag upstream, or null when the repo has none. */
  latestTag: string | null
  behind: boolean
  /** False when the remote could not be reached; the card shows nothing rather than "up to date". */
  checked: boolean
}

export interface RegistryIndex {
  entries: RegistryEntry[]
  /** True when the fetch failed and these entries came from the on-disk cache. */
  stale: boolean
}

export type InstallPhase =
  | 'fetch'
  | 'validate'
  | 'scan'
  | 'preflight'
  | 'resolve'
  | 'install'
  | 'lock'
  | 'register'
  | 'activate'
  | 'done'

export interface InstallProgressEvent {
  phase: InstallPhase
  fraction: number
  status: string
}

/** A successful (or consent-paused) install. */
export interface InstallSuccess {
  ok: true
  extensionId: string
  version: string
  name: string
  nodeTypes: string[]
  scan: ScanReport | null
  /** The extension was already imported this session, so the new code is not live until a restart. */
  restartRequired: boolean
  /** Paused for consent: nothing was installed. Re-call with the report's `consentRules`. */
  needsConsent: boolean
}

export interface InstallFailure {
  ok: false
  error: string
  phase: InstallPhase
  conflicts: DependencyConflict[]
  scan: ScanReport | null
  restartRequired: boolean
}

export type InstallOutcome = InstallSuccess | InstallFailure

export interface LifecycleResult {
  extensionId: string
  restartRequired: boolean
}

/** Human-readable label for an install phase, for the progress line. */
export const PHASE_LABELS: Record<InstallPhase, string> = {
  fetch: 'Downloading',
  validate: 'Checking the manifest',
  scan: 'Reviewing the code',
  preflight: 'Checking for conflicts',
  resolve: 'Resolving dependencies',
  install: 'Installing dependencies',
  lock: 'Recording the install',
  register: 'Loading nodes',
  activate: 'Activating',
  done: 'Installed',
}

export const SEVERITY_LABELS: Record<FindingSeverity, string> = {
  critical: 'Blocked',
  high: 'Needs your approval',
  medium: 'Needs your approval',
  low: 'For your information',
}

/** True when this node type came from an extension rather than Core. */
export function isExtensionNode(source: string): boolean {
  return source.startsWith('ext:')
}

/** The owning extension id of an `ext:<extension>:<module>` source, or null. */
export function extensionOf(source: string): string | null {
  const parts = source.split(':')
  return parts[0] === 'ext' && parts[1] ? parts[1] : null
}
