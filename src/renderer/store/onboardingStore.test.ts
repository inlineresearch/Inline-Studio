import { beforeEach, describe, expect, it } from 'vitest'

import { useOnboardingStore } from './onboardingStore'

const KEY = 'inline-studio.onboarding.starterHints'
const target = { itemId: 'n1', surface: 'studio' as const }

/**
 * The suite runs in node with no DOM, which is the repo's default and worth keeping: a store that
 * only touches localStorage behind try/catch should not drag jsdom into every other test. So the
 * shim is local, and one case removes it entirely to prove the guards actually hold.
 */
function installStorage(over: Partial<Storage> = {}): Map<string, string> {
  const map = new Map<string, string>()
  const store: Partial<Storage> = {
    getItem: (k: string) => map.get(k) ?? null,
    setItem: (k: string, v: string) => void map.set(k, v),
    removeItem: (k: string) => void map.delete(k),
    clear: () => map.clear(),
    ...over,
  }
  Object.defineProperty(globalThis, 'localStorage', { value: store, configurable: true })
  return map
}

function removeStorage(): void {
  Reflect.deleteProperty(globalThis as Record<string, unknown>, 'localStorage')
}

describe('onboardingStore', () => {
  beforeEach(() => {
    installStorage()
    useOnboardingStore.setState({ starterHintsSeen: false, hintTarget: null })
  })

  it('marks seen when the hint is shown, not when it is dismissed', () => {
    // A reload part-way through the hint should not replay it.
    const map = installStorage()
    useOnboardingStore.getState().armHints(target)
    expect(map.get(KEY)).toBe('1')
    expect(useOnboardingStore.getState().hintTarget).toEqual(target)

    useOnboardingStore.getState().dismissHints()
    expect(useOnboardingStore.getState().hintTarget).toBeNull()
    expect(useOnboardingStore.getState().starterHintsSeen).toBe(true)
  })

  it('will not re-arm once seen', () => {
    useOnboardingStore.getState().armHints(target)
    useOnboardingStore.getState().dismissHints()
    useOnboardingStore.getState().armHints({ itemId: 'n2' })
    expect(useOnboardingStore.getState().hintTarget).toBeNull()
  })

  it('still shows the hint when localStorage throws', () => {
    // Private browsing. Not remembering the hint is survivable; crashing the canvas is not.
    installStorage({
      setItem: () => {
        throw new Error('quota exceeded')
      },
    })
    expect(() => useOnboardingStore.getState().armHints(target)).not.toThrow()
    expect(useOnboardingStore.getState().hintTarget).toEqual(target)
  })

  it('survives localStorage being absent entirely', () => {
    removeStorage()
    expect(() => useOnboardingStore.getState().armHints(target)).not.toThrow()
    expect(() => useOnboardingStore.getState().resetSeen()).not.toThrow()
    installStorage()
  })

  it('can be reset, which is the seam for a replay action', () => {
    const map = installStorage()
    useOnboardingStore.getState().armHints(target)
    useOnboardingStore.getState().resetSeen()
    expect(map.get(KEY)).toBeUndefined()
    expect(useOnboardingStore.getState().starterHintsSeen).toBe(false)
  })
})
