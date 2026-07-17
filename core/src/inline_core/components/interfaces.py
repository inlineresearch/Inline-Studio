"""The five component interfaces. Device- and dtype-agnostic; placement comes from the context.

The Sampler carries an on_step callback from day one so progress streams without retrofitting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .conditioning import Conditioning, Latents

if TYPE_CHECKING:
    import torch

    from ..runtime.context import ExecutionContext


@dataclass(frozen=True)
class StepInfo:
    """Reported once per denoise step so the executor can stream sample progress."""

    step: int
    total: int


StepCallback = Callable[[StepInfo], None]


class TextEncoder(ABC):
    """prompt -> opaque Conditioning."""

    @abstractmethod
    def encode(
        self, prompt: str, params: dict[str, Any], ctx: ExecutionContext
    ) -> Conditioning: ...


class Scheduler(ABC):
    """Owns the timesteps and the prediction parameterization."""

    @abstractmethod
    def timesteps(self, steps: int) -> Sequence[torch.Tensor]: ...

    @abstractmethod
    def scale_model_input(self, latents: Latents, t: torch.Tensor) -> Latents: ...

    @abstractmethod
    def step(self, model_output: torch.Tensor, t: torch.Tensor, latents: Latents) -> Latents: ...


class Denoiser(ABC):
    """(latents, timestep, conditioning) -> predicted output. The heavy model."""

    @abstractmethod
    def predict(
        self,
        latents: Latents,
        timestep: torch.Tensor,
        conditioning: Conditioning,
        ctx: ExecutionContext,
    ) -> torch.Tensor: ...


class Sampler(ABC):
    """The stepping loop, independent of the denoiser. Reports each step via on_step."""

    @abstractmethod
    def sample(
        self,
        denoiser: Denoiser,
        scheduler: Scheduler,
        latents: Latents,
        conditioning: Conditioning,
        *,
        steps: int,
        ctx: ExecutionContext,
        on_step: StepCallback | None = None,
    ) -> Latents: ...


class VAE(ABC):
    """Pixel and latent conversion, both directions."""

    @abstractmethod
    def encode(self, image: torch.Tensor, ctx: ExecutionContext) -> Latents: ...

    @abstractmethod
    def decode(self, latents: Latents, ctx: ExecutionContext) -> torch.Tensor: ...
