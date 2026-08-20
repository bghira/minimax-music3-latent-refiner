from __future__ import annotations

import types

import torch

from .pipeline import MiniMaxMusic3RefinerPipeline, RefinerOutput


def attach_latent_refiner(diffusers_pipeline, refiner: MiniMaxMusic3RefinerPipeline):
    """Attach `refine_audio` to a loaded MiniMax Music 3 Diffusers pipeline instance."""
    if not hasattr(diffusers_pipeline, "vocoder") or diffusers_pipeline.vocoder is None:
        raise TypeError("the Diffusers pipeline must expose its MiniMax Music 3 vocoder")
    if hasattr(diffusers_pipeline, "refine_audio"):
        raise ValueError("the Diffusers pipeline already has a refine_audio attribute")

    def refine_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        steps: int = 32,
        window_seconds: float | None = 30.0,
        overlap_seconds: float = 2.0,
    ) -> RefinerOutput:
        return refiner(
            waveform,
            sample_rate,
            steps=steps,
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
            decoder=self.vocoder,
        )

    diffusers_pipeline.refine_audio = types.MethodType(refine_audio, diffusers_pipeline)
    return diffusers_pipeline
