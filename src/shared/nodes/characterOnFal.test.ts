import { describe, expect, it } from 'vitest'
import { MINIMAX_H3_REF2V } from './minimaxH3Ref2V'
import { NANO_BANANA_PRO } from './nanoBananaPro'
import { SEEDANCE_REF2V } from './seedanceRef2V'
import { emptyResolvedInputs, type AppliedCharacter, type ResolvedInputs } from './types'

const uri = (n: string): string => `data:image/png;base64,${n}`

function applied(refs: string[], promptPrefix = ''): AppliedCharacter {
  return { name: 'Emmy', refs, roles: refs.map(() => 'face'), promptPrefix }
}

function inputs(over: Partial<ResolvedInputs> = {}): ResolvedInputs {
  return { ...emptyResolvedInputs(), ...over }
}

describe('a character on Nano Banana Pro', () => {
  it('appends its references after the wired ones', () => {
    // Position is what the prompt resolves, and the numbers on the node face come from the wires
    // alone - leading the character would shift every number the user can see.
    const body = NANO_BANANA_PRO.buildRequest(
      { prompt: 'on a beach' },
      inputs({ images: [uri('wired')], character: applied([uri('a'), uri('b')]) }),
    )
    expect(body.image_urls).toEqual([uri('wired'), uri('a'), uri('b')])
  })

  it('prefixes the prompt with the binding text Core composed', () => {
    const body = NANO_BANANA_PRO.buildRequest(
      { prompt: 'on a beach' },
      inputs({ images: [uri('wired')], character: applied([uri('a')], 'Image 2 shows Emmy. ') }),
    )
    expect(body.prompt).toBe('Image 2 shows Emmy. on a beach')
  })

  it('runs on the character alone, with nothing wired', () => {
    const body = NANO_BANANA_PRO.buildRequest(
      { prompt: 'on a beach' },
      inputs({ character: applied([uri('a')]) }),
    )
    expect(body.image_urls).toEqual([uri('a')])
  })

  it('reads ordinal prose, so it declares no @Image style', () => {
    // Nano Banana takes ordinary prose; handing it Seedance's syntax would name nothing.
    expect(NANO_BANANA_PRO.character?.style).toBe('ordinal')
  })

  it('claims at most the 5 slots documented for character consistency', () => {
    // The port takes 14 references, but only 5 of them are the identity band Google documents.
    expect(NANO_BANANA_PRO.character?.maxRefs).toBe(5)
    expect(NANO_BANANA_PRO.character?.maxImages).toBe(14)
  })
})

describe('a character on Seedance Reference to Video', () => {
  it('uses the @Image syntax the endpoint documents', () => {
    expect(SEEDANCE_REF2V.character?.style).toBe('at-image')
  })

  it('appends its references after the wired ones', () => {
    const body = SEEDANCE_REF2V.buildRequest(
      { prompt: 'walking' },
      inputs({ images: [uri('wired')], character: applied([uri('a')], '@Image2 shows Emmy. ') }),
    )
    expect(body.image_urls).toEqual([uri('wired'), uri('a')])
    expect(body.prompt).toBe('@Image2 shows Emmy. walking')
  })
})

describe('Seedance reference caps', () => {
  it('trims to the 9 images fal accepts rather than letting the API 422', () => {
    const many = Array.from({ length: 12 }, (_, i) => uri(`w${i}`))
    const body = SEEDANCE_REF2V.buildRequest({ prompt: 'x' }, inputs({ images: many }))
    expect((body.image_urls as string[]).length).toBe(9)
  })

  it('counts the character against the same 9, never past it', () => {
    const wired = Array.from({ length: 7 }, (_, i) => uri(`w${i}`))
    const body = SEEDANCE_REF2V.buildRequest(
      { prompt: 'x' },
      inputs({ images: wired, character: applied([uri('a'), uri('b'), uri('c'), uri('d')]) }),
    )
    // The wired seven keep their positions; the character fills what is left.
    expect(body.image_urls).toEqual([...wired, uri('a'), uri('b')])
  })

  it('drops audio first when the 12-file combined cap bites', () => {
    // Audio is the input the model can do without; losing a reference image loses the identity.
    const images = Array.from({ length: 9 }, (_, i) => uri(`i${i}`))
    const body = SEEDANCE_REF2V.buildRequest(
      { prompt: 'x' },
      inputs({ images, videos: [uri('v1'), uri('v2'), uri('v3')], audios: [uri('a1')] }),
    )
    expect((body.image_urls as string[]).length).toBe(9)
    expect((body.video_urls as string[]).length).toBe(3)
    expect(body.audio_urls).toBeUndefined()
  })

  it('leaves an absent input off the body entirely', () => {
    // fal reads an empty array as a real (invalid) value, not as "unset".
    const body = SEEDANCE_REF2V.buildRequest({ prompt: 'x' }, inputs({ images: [uri('a')] }))
    expect('video_urls' in body).toBe(false)
    expect('audio_urls' in body).toBe(false)
  })
})

describe('a character on MiniMax H3 Reference to Video', () => {
  it('reads `<Picture N>` tokens, which is the only form H3 resolves', () => {
    // Handing it Seedance's `@ImageN` or FLUX.2's prose names positions H3 cannot see.
    expect(MINIMAX_H3_REF2V.character?.style).toBe('token')
  })

  it('appends its references after the wired ones, into the reference port', () => {
    const body = MINIMAX_H3_REF2V.buildRequest(
      { prompt: 'walking' },
      inputs({
        byHandle: { reference_image_urls: [uri('wired')] },
        character: applied([uri('a'), uri('b')], '<Picture 2> <Picture 3> show Emmy. '),
      }),
    )
    expect(body.reference_image_urls).toEqual([uri('wired'), uri('a'), uri('b')])
    expect(body.prompt).toBe('<Picture 2> <Picture 3> show Emmy. walking')
  })

  it('accepts every role, unlike Seedance', () => {
    // The whole reason this node carries a character: measured, it takes the face references.
    expect(MINIMAX_H3_REF2V.character?.excludeRoles).toBeUndefined()
  })
})
