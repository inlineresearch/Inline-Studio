/**
 * Turn anything thrown by an IPC call into a readable message. Guards against the
 * "silent nothing happens" failure mode where a missing/old `studio()`
 * method throws and the rejection is discarded — surface it in the UI instead.
 */
export function ipcErrorMessage(e: unknown): string {
  if (e instanceof Error) {
    if (e.message.includes('is not a function') || e.message.includes('undefined')) {
      return 'This action is unavailable — please fully restart the app (the bridge is out of date).'
    }
    return e.message
  }
  return String(e)
}
