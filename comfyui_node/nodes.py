from __future__ import annotations

import comfy.model_management
import torch

from minimax_music3_latent_refiner import MiniMaxMusic3DAV, MiniMaxMusic3RefinerPipeline
from minimax_music3_latent_refiner.pipeline import DEFAULT_AUDIO_VAE_ID, DEFAULT_MODEL_ID

DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


class MiniMaxMusic3LatentRefinerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {"default": DEFAULT_MODEL_ID}),
                "precision": (list(DTYPES), {"default": "fp32"}),
            }
        }

    RETURN_TYPES = ("MINIMAX_MUSIC3_REFINER", "AUDIO_VAE_ENCODER")
    RETURN_NAMES = ("refiner", "bundled_audio_vae_encoder")
    FUNCTION = "load"
    CATEGORY = "audio/minimax music3"

    def load(self, model_id: str, precision: str):
        pipeline = MiniMaxMusic3RefinerPipeline.from_pretrained(
            model_id,
            device=comfy.model_management.get_torch_device(),
            dtype=DTYPES[precision],
        )
        return pipeline, pipeline.audio_vae


class MiniMaxMusic3AudioVAEEncoderLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {"default": DEFAULT_AUDIO_VAE_ID}),
                "precision": (list(DTYPES), {"default": "fp32"}),
            }
        }

    RETURN_TYPES = ("AUDIO_VAE_ENCODER",)
    RETURN_NAMES = ("audio_vae_encoder",)
    FUNCTION = "load"
    CATEGORY = "audio/minimax music3"

    def load(self, model_id: str, precision: str):
        encoder = MiniMaxMusic3DAV.from_pretrained(
            model_id,
            subfolder="audio_vae",
            torch_dtype=DTYPES[precision],
        ).to(comfy.model_management.get_torch_device())
        encoder.eval().requires_grad_(False)
        return (encoder,)


class MiniMaxMusic3LatentRefine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refiner": ("MINIMAX_MUSIC3_REFINER",),
                "audio": ("AUDIO",),
                "steps": ("INT", {"default": 32, "min": 1, "max": 128}),
                "window_seconds": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 600.0, "step": 0.5}),
                "overlap_seconds": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 30.0, "step": 0.25}),
                "direct_sequence": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "audio_vae_encoder": ("AUDIO_VAE_ENCODER",),
                "audio_vae_decoder": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT", "AUDIO")
    RETURN_NAMES = ("refined_latents", "refined_audio")
    FUNCTION = "refine"
    CATEGORY = "audio/minimax music3"

    def refine(
        self,
        refiner: MiniMaxMusic3RefinerPipeline,
        audio: dict,
        steps: int,
        window_seconds: float,
        overlap_seconds: float,
        direct_sequence: bool,
        audio_vae_encoder=None,
        audio_vae_decoder=None,
    ):
        if audio is None or "waveform" not in audio or "sample_rate" not in audio:
            raise ValueError("audio must be a ComfyUI AUDIO value")
        latents, original_samples = refiner.refine_latents(
            audio["waveform"],
            int(audio["sample_rate"]),
            steps=steps,
            window_seconds=None if direct_sequence else window_seconds,
            overlap_seconds=overlap_seconds,
            audio_encoder=audio_vae_encoder,
        )
        waveform = refiner.decode_latents(latents, original_samples, decoder=audio_vae_decoder)
        latent_output = {
            "samples": latents.to(comfy.model_management.intermediate_device()),
            "sample_rate": 44_100,
        }
        audio_output = {"waveform": waveform, "sample_rate": 44_100}
        return latent_output, audio_output


NODE_CLASS_MAPPINGS = {
    "MiniMaxMusic3LatentRefinerLoader": MiniMaxMusic3LatentRefinerLoader,
    "MiniMaxMusic3AudioVAEEncoderLoader": MiniMaxMusic3AudioVAEEncoderLoader,
    "MiniMaxMusic3LatentRefine": MiniMaxMusic3LatentRefine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxMusic3LatentRefinerLoader": "Load MiniMax Music3 Latent Refiner",
    "MiniMaxMusic3AudioVAEEncoderLoader": "Load MiniMax Music3 Audio VAE Encoder",
    "MiniMaxMusic3LatentRefine": "MiniMax Music3 Latent Refine",
}
