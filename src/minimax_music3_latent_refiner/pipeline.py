from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel, ClapModel, ClapProcessor, Wav2Vec2FeatureExtractor

from .dav import MiniMaxMusic3DAV
from .model import MiniMaxMusic3LatentRefiner, bridge_sample

SAMPLE_RATE = 44_100
MERT_SAMPLE_RATE = 24_000
DAV_HOP = 512
DEFAULT_MODEL_ID = "terminusresearch/minimax-music3-latent-refiner-v0.10"
DEFAULT_AUDIO_VAE_ID = "SimpleTuner/MiniMax-Music-3-Encoder"
DEFAULT_MERT_ID = "m-a-p/MERT-v1-95M"
DEFAULT_CLAP_ID = "laion/larger_clap_music"


class AudioVAEEncoder(Protocol):
    def encode(self, waveform: torch.Tensor) -> torch.Tensor: ...


@dataclass
class RefinerOutput:
    latents: torch.Tensor
    audio: torch.Tensor | None
    sample_rate: int
    original_samples: int


def _module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


def prepare_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if waveform.ndim == 1:
        waveform = waveform[None, None]
    elif waveform.ndim == 2:
        waveform = waveform[None]
    if waveform.ndim != 3:
        raise ValueError("waveform must have shape [samples], [channels, samples], or [batch, channels, samples]")
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    elif waveform.shape[1] != 2:
        raise ValueError("waveform must be mono or stereo")
    waveform = waveform.float().cpu()
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    return waveform


def window_starts(total_samples: int, window_samples: int, overlap_samples: int) -> list[int]:
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")
    if not 0 <= overlap_samples < window_samples:
        raise ValueError("overlap_samples must be non-negative and smaller than window_samples")
    if total_samples <= window_samples:
        return [0]
    stride = window_samples - overlap_samples
    starts = list(range(0, total_samples - window_samples + 1, stride))
    final_start = total_samples - window_samples
    final_start -= final_start % DAV_HOP
    if final_start > starts[-1]:
        starts.append(final_start)
    return starts


class MiniMaxMusic3RefinerPipeline:
    def __init__(
        self,
        refiner: MiniMaxMusic3LatentRefiner,
        audio_vae: MiniMaxMusic3DAV,
        mert: torch.nn.Module,
        mert_processor: Wav2Vec2FeatureExtractor,
        clap: ClapModel,
        clap_processor: ClapProcessor,
        normalization: dict[str, torch.Tensor],
    ):
        required = {"latent_mean", "latent_std", "layer_mean", "layer_std"}
        missing = required.difference(normalization)
        if missing:
            raise ValueError(f"normalization is missing tensors: {sorted(missing)}")
        self.refiner = refiner.eval()
        self.audio_vae = audio_vae.eval()
        self.mert = mert.eval()
        self.mert_processor = mert_processor
        self.clap = clap.eval()
        self.clap_processor = clap_processor
        self.normalization = normalization

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | Path = DEFAULT_MODEL_ID,
        *,
        audio_vae_id: str = DEFAULT_AUDIO_VAE_ID,
        mert_id: str = DEFAULT_MERT_ID,
        clap_id: str = DEFAULT_CLAP_ID,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> MiniMaxMusic3RefinerPipeline:
        device = torch.device(device)
        model_id_string = str(model_id)
        refiner = MiniMaxMusic3LatentRefiner.from_pretrained(
            model_id_string,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        ).to(device)
        if Path(model_id_string).is_dir():
            normalization_path = Path(model_id_string) / "normalization.safetensors"
        else:
            normalization_path = Path(
                hf_hub_download(
                    model_id_string,
                    "normalization.safetensors",
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
            )
        if not normalization_path.is_file():
            raise FileNotFoundError(f"required normalization file not found: {normalization_path}")
        normalization = load_file(normalization_path, device=str(device))
        audio_vae = MiniMaxMusic3DAV.from_pretrained(
            audio_vae_id,
            subfolder="audio_vae",
            torch_dtype=dtype,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        ).to(device)
        mert = AutoModel.from_pretrained(
            mert_id,
            trust_remote_code=True,
            dtype=dtype,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        ).to(device)
        mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            mert_id,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        clap = ClapModel.from_pretrained(
            clap_id,
            dtype=dtype,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        ).to(device)
        clap_processor = ClapProcessor.from_pretrained(
            clap_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        refiner.requires_grad_(False)
        audio_vae.requires_grad_(False)
        mert.requires_grad_(False)
        clap.requires_grad_(False)
        return cls(refiner, audio_vae, mert, mert_processor, clap, clap_processor, normalization)

    @property
    def device(self) -> torch.device:
        return _module_device(self.refiner)

    @property
    def dtype(self) -> torch.dtype:
        return _module_dtype(self.refiner)

    def _normalization_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device
        return (
            self.normalization["latent_mean"].to(device)[None, None],
            self.normalization["latent_std"].to(device)[None, None],
            self.normalization["layer_mean"].to(device)[None, :, None],
            self.normalization["layer_std"].to(device)[None, :, None],
        )

    @torch.inference_mode()
    def _mert_layers(self, waveform: torch.Tensor, frame_count: int) -> torch.Tensor:
        mono = waveform.mean(dim=1)
        mono_24k = torchaudio.functional.resample(mono, SAMPLE_RATE, MERT_SAMPLE_RATE)
        inputs = self.mert_processor(
            [row.numpy() for row in mono_24k],
            sampling_rate=MERT_SAMPLE_RATE,
            return_tensors="pt",
        )
        outputs = self.mert(inputs["input_values"].to(self.device), output_hidden_states=True)
        layers = torch.stack(outputs.hidden_states, dim=1).float()
        batch, layer_count, mert_frames, dim = layers.shape
        return (
            F.interpolate(
                layers.reshape(batch * layer_count, mert_frames, dim).transpose(1, 2),
                size=frame_count,
                mode="linear",
                align_corners=True,
            )
            .transpose(1, 2)
            .reshape(batch, layer_count, frame_count, dim)
        )

    @torch.inference_mode()
    def _clap_style(self, waveform: torch.Tensor) -> torch.Tensor:
        mono = waveform.mean(dim=1)
        center = mono.shape[-1] // 2
        half = min(5 * SAMPLE_RATE, center)
        clip = mono[..., center - half : center + half]
        clip = torchaudio.functional.resample(clip, SAMPLE_RATE, 48_000)
        inputs = self.clap_processor(
            audio=[row.numpy() for row in clip],
            sampling_rate=48_000,
            return_tensors="pt",
        )
        features = self.clap.get_audio_features(input_features=inputs["input_features"].to(self.device))
        if not torch.is_tensor(features):
            features = features.pooler_output
        if features.shape[-1] != self.clap.config.projection_dim:
            features = self.clap.audio_projection(features)
        return features.to(device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def _refine_chunk(
        self,
        waveform: torch.Tensor,
        steps: int,
        audio_encoder: AudioVAEEncoder,
    ) -> torch.Tensor:
        encoder_device = _module_device(audio_encoder) if isinstance(audio_encoder, torch.nn.Module) else self.device
        encoder_dtype = _module_dtype(audio_encoder) if isinstance(audio_encoder, torch.nn.Module) else self.dtype
        degraded = audio_encoder.encode(waveform.to(device=encoder_device, dtype=encoder_dtype))
        if degraded.ndim != 3 or degraded.shape[1] != self.refiner.config.latent_dim:
            raise ValueError("AUDIO_VAE_ENCODER.encode must return [batch, 128, frames]")
        degraded = degraded.transpose(1, 2).to(device=self.device, dtype=self.dtype)
        layers = self._mert_layers(waveform, degraded.shape[1])
        style = self._clap_style(waveform)
        latent_mean, latent_std, layer_mean, layer_std = self._normalization_tensors()
        degraded = (degraded - latent_mean) / latent_std
        layers = ((layers - layer_mean) / layer_std).to(self.dtype)
        generated = bridge_sample(
            self.refiner,
            layers[:, self.refiner.config.mert_input_layer],
            layers,
            degraded,
            style,
            steps,
        )
        return (generated * latent_std + latent_mean).transpose(1, 2)

    @torch.inference_mode()
    def refine_latents(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        steps: int = 32,
        window_seconds: float | None = 30.0,
        overlap_seconds: float = 2.0,
        audio_encoder: AudioVAEEncoder | None = None,
    ) -> tuple[torch.Tensor, int]:
        waveform = prepare_waveform(waveform, sample_rate)
        original_samples = waveform.shape[-1]
        encoder = audio_encoder if audio_encoder is not None else self.audio_vae
        if not hasattr(encoder, "encode"):
            raise TypeError("audio_encoder must provide encode(waveform) -> [batch, 128, frames]")
        if window_seconds is None:
            return self._refine_chunk(waveform, steps, encoder), original_samples
        if waveform.shape[0] != 1 and waveform.shape[-1] > round(window_seconds * SAMPLE_RATE):
            raise ValueError("windowed refinement currently requires batch size 1")

        window_samples = max(DAV_HOP, round(window_seconds * SAMPLE_RATE / DAV_HOP) * DAV_HOP)
        overlap_samples = max(0, round(overlap_seconds * SAMPLE_RATE / DAV_HOP) * DAV_HOP)
        starts = window_starts(original_samples, window_samples, overlap_samples)
        if len(starts) == 1:
            return self._refine_chunk(waveform, steps, encoder), original_samples

        total_frames = math.ceil(original_samples / DAV_HOP)
        accumulated = torch.zeros(1, 128, total_frames, device=self.device, dtype=self.dtype)
        weights = torch.zeros(1, 1, total_frames, device=self.device, dtype=self.dtype)
        chunk_frames = math.ceil(window_samples / DAV_HOP)
        for index, start_sample in enumerate(starts):
            chunk = waveform[..., start_sample : start_sample + window_samples]
            generated = self._refine_chunk(chunk, steps, encoder)
            start_frame = start_sample // DAV_HOP
            available = min(generated.shape[-1], total_frames - start_frame)
            blend = torch.ones(available, device=self.device, dtype=self.dtype)
            if index > 0:
                previous_end = starts[index - 1] // DAV_HOP + chunk_frames
                left = min(available, max(0, previous_end - start_frame))
                if left:
                    blend[:left] = torch.linspace(0.0, 1.0, left + 2, device=self.device, dtype=self.dtype)[1:-1]
            if index + 1 < len(starts):
                next_start = starts[index + 1] // DAV_HOP
                right = min(available, max(0, start_frame + available - next_start))
                if right:
                    fade = torch.linspace(1.0, 0.0, right + 2, device=self.device, dtype=self.dtype)[1:-1]
                    blend[-right:] = torch.minimum(blend[-right:], fade)
            accumulated[..., start_frame : start_frame + available] += generated[..., :available] * blend
            weights[..., start_frame : start_frame + available] += blend
        if torch.any(weights == 0):
            raise RuntimeError("window overlap left uncovered latent frames")
        return accumulated / weights, original_samples

    @torch.inference_mode()
    def decode_latents(
        self,
        latents: torch.Tensor,
        original_samples: int,
        decoder: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        decoder = decoder if decoder is not None else self.audio_vae
        if isinstance(decoder, torch.nn.Module):
            decoder_input = latents.to(device=_module_device(decoder), dtype=_module_dtype(decoder))
        else:
            decoder_input = latents.cpu()
        if hasattr(decoder, "decode"):
            audio = decoder.decode(decoder_input)
        else:
            audio = decoder(decoder_input)
        if audio.ndim != 3:
            raise ValueError("audio decoder must return a rank-3 waveform tensor")
        if audio.shape[1] != 2 and audio.shape[-1] == 2:
            audio = audio.movedim(-1, 1)
        if audio.shape[1] != 2:
            raise ValueError("audio decoder must return stereo [batch, 2, samples]")
        return audio[..., :original_samples].float().cpu()

    def __call__(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        steps: int = 32,
        window_seconds: float | None = 30.0,
        overlap_seconds: float = 2.0,
        audio_encoder: AudioVAEEncoder | None = None,
        decoder: torch.nn.Module | None = None,
        decode: bool = True,
    ) -> RefinerOutput:
        latents, original_samples = self.refine_latents(
            waveform,
            sample_rate,
            steps=steps,
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
            audio_encoder=audio_encoder,
        )
        audio = self.decode_latents(latents, original_samples, decoder) if decode else None
        return RefinerOutput(latents, audio, SAMPLE_RATE, original_samples)
