"""Krea 2 img2img: start the schedule partway, from an encoded input image.

diffusers ships no ``Krea2Img2ImgPipeline``, but ``Krea2Pipeline.__call__`` already accepts
pre-noised ``latents`` and an explicit ``sigmas`` schedule, so img2img needs no fork of the denoise
loop - only the right latents and the right tail of the schedule.
"""

from __future__ import annotations

from typing import Any

import torch


def img2img_kwargs(
    pipe: Any,
    *,
    image: Any,
    strength: float,
    steps: int,
    width: int,
    height: int,
    generator: Any,
    device: str,
) -> dict[str, Any]:
    """Start the schedule partway, from the input image, using the pipeline's own ``latents`` and
    ``sigmas`` inputs - diffusers ships no ``Krea2Img2ImgPipeline`` and forking its denoise loop
    would be a maintenance liability.

    The sigma tail is handed over **unshifted**: ``set_timesteps`` applies the same monotone
    exponential shift the pipeline would, so a shifted subset and a subset of the shifted schedule
    are the same thing. The latents are noised at the *shifted* start sigma so the noise level the
    model sees matches the timestep it is given."""
    import numpy as np
    from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift

    raw = np.linspace(1.0, 1 / steps, steps)
    start = min(steps - 1, max(0, int(round((1.0 - strength) * steps))))

    vae_scale = pipe.vae_scale_factor * pipe.patch_size
    grid_h, grid_w = height // vae_scale, width // vae_scale
    if pipe.config.is_distilled:
        mu = 1.15
    else:
        config = pipe.scheduler.config
        mu = calculate_shift(
            grid_h * grid_w,
            config.get("base_image_seq_len", 256),
            config.get("max_image_seq_len", 6400),
            config.get("base_shift", 0.5),
            config.get("max_shift", 1.15),
        )
    schedule = pipe.scheduler.__class__.from_config(pipe.scheduler.config)
    schedule.set_timesteps(sigmas=raw, mu=mu, device=device)
    sigma = schedule.sigmas[start].to(device)

    latents = encode_image(pipe, image, width, height, device, generator)
    noise = torch.randn(
        latents.shape, generator=generator, device=latents.device, dtype=latents.dtype
    )
    return {"latents": (1.0 - sigma) * latents + sigma * noise, "sigmas": raw[start:].tolist()}


def encode_image(
    pipe: Any, image: Any, width: int, height: int, device: str, generator: Any
) -> Any:
    """The input image as packed, normalized Krea 2 latents. The Qwen-Image VAE is a video codec, so
    the pixels carry a length-1 temporal axis. The VAE sample is drawn from the run's seeded
    generator, not the global RNG, so the same seed reproduces the same img2img result."""
    import numpy as np

    resized = image.convert("RGB").resize((width, height))
    array = np.asarray(resized, dtype="float32") / 127.5 - 1.0
    dtype = pipe.vae.dtype
    pixels = torch.from_numpy(array).permute(2, 0, 1)[None, :, None].to(device, dtype)

    with torch.no_grad():
        latents = pipe.vae.encode(pixels).latent_dist.sample(generator=generator)
    mean, std = _latent_stats(pipe.vae, latents)
    latents = ((latents - mean) / std).squeeze(2)  # drop the temporal axis
    batch, channels, latent_h, latent_w = latents.shape
    return pipe._pack_latents(latents, batch, channels, latent_h, latent_w)


def _latent_stats(vae: Any, latents: Any) -> tuple[Any, Any]:
    shape = (1, vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
    std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype)
    return mean.view(shape), std.view(shape)
