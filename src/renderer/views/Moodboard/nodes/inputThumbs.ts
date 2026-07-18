import { takeWaveformPath } from '@shared/media'
import type { Asset, Frame, FrameInput, Take } from '@shared/types'
import { resolveMedia } from '@/lib/media'

/** A resolved input thumbnail - from a library asset OR a flow/source-frame input. */
export interface InputThumb {
  /** The frame_input row id (stable key; used to remove the input). */
  id: string
  /** The library asset id, or null for a flow (source-frame) input. */
  assetId: string | null
  /** The media URL to display (downscaled thumb / playable transcode). */
  url: string
  /** The original media to save on right-click (not the transcoded video preview). */
  saveSrc: string
  kind: 'image' | 'video' | 'audio'
  /** Poster still for a video, so it renders even when the codec can't be decoded. */
  poster?: string
  /** Waveform peaks JSON URL, for audio inputs/takes. */
  waveform?: string
}

/** The store slices needed to resolve a frame's inputs to displayable thumbnails. */
export interface InputThumbCtx {
  assets: Asset[]
  allFrames: Frame[]
  takesByFrame: Record<string, Take[]>
  inputsByFrame: Record<string, FrameInput[]>
}

/**
 * Resolve a frame's inputs to thumbnails, in order. Asset inputs map to their library media; flow
 * inputs (`sourceFrameId`, from a wired frame/Preview) map to that frame's hero take - or, when it
 * hasn't generated yet, its own imported input asset. Shared by the Frame and Generate nodes so
 * both surface their inputs the same way. Rows that don't resolve are dropped.
 */
export function resolveInputThumbs(inputs: FrameInput[], ctx: InputThumbCtx): InputThumb[] {
  const { assets, allFrames, takesByFrame, inputsByFrame } = ctx
  return inputs
    .map((i): InputThumb | null => {
      if (i.assetId) {
        const a = assets.find((x) => x.id === i.assetId)
        if (!a) return null
        return {
          id: i.id,
          assetId: a.id,
          url: resolveMedia(
            a.kind === 'image' ? (a.thumbPath ?? a.filePath) : (a.previewPath ?? a.filePath),
          ),
          saveSrc: resolveMedia(a.filePath),
          kind: a.kind,
          poster: a.kind === 'video' && a.thumbPath ? resolveMedia(a.thumbPath) : undefined,
          waveform: a.kind === 'audio' && a.thumbPath ? resolveMedia(a.thumbPath) : undefined,
        }
      }
      if (i.sourceFrameId) {
        const sf = allFrames.find((f) => f.id === i.sourceFrameId)
        const takes = sf ? (takesByFrame[sf.id] ?? []) : []
        // Mirror the Preview: the hero take, or the newest when no hero is set.
        const take = takes.find((t) => t.id === sf?.heroTakeId) ?? takes[0]
        if (take) {
          return {
            id: i.id,
            assetId: null,
            url: resolveMedia(take.filePath),
            saveSrc: resolveMedia(take.filePath),
            kind: take.kind,
            waveform: take.kind === 'audio' ? resolveMedia(takeWaveformPath(take.id)) : undefined,
          }
        }
        // No take yet - fall back to the source frame's imported input asset.
        const srcInput = sf ? (inputsByFrame[sf.id] ?? []).find((x) => x.assetId) : undefined
        const srcAsset = srcInput?.assetId
          ? assets.find((a) => a.id === srcInput.assetId)
          : undefined
        return srcAsset
          ? {
              id: i.id,
              assetId: null,
              url: resolveMedia(srcAsset.previewPath ?? srcAsset.filePath),
              saveSrc: resolveMedia(srcAsset.filePath),
              kind: srcAsset.kind,
              poster:
                srcAsset.kind === 'video' && srcAsset.thumbPath
                  ? resolveMedia(srcAsset.thumbPath)
                  : undefined,
              waveform:
                srcAsset.kind === 'audio' && srcAsset.thumbPath
                  ? resolveMedia(srcAsset.thumbPath)
                  : undefined,
            }
          : null
      }
      return null
    })
    .filter((t): t is InputThumb => !!t)
}
