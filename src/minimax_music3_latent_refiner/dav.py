# Copyright 2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import math

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch import nn
from torch.nn.utils import weight_norm


class MiniMaxMusic3Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
        hidden_states = hidden_states + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * hidden_states).pow(2)
        return hidden_states.reshape(shape)


class DAVResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            MiniMaxMusic3Snake1d(dim),
            weight_norm(nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=3 * dilation)),
            MiniMaxMusic3Snake1d(dim),
            weight_norm(nn.Conv1d(dim, dim, kernel_size=1)),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.block(hidden_states)
        if residual.shape[-1] != hidden_states.shape[-1]:
            padding = (hidden_states.shape[-1] - residual.shape[-1]) // 2
            hidden_states = hidden_states[..., padding : hidden_states.shape[-1] - padding]
        return hidden_states + residual


class DAVEncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            DAVResidualUnit(dim // 2, dilation=1),
            DAVResidualUnit(dim // 2, dilation=3),
            DAVResidualUnit(dim // 2, dilation=9),
            MiniMaxMusic3Snake1d(dim // 2),
            weight_norm(nn.Conv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2))),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states)


class DAVEncoder(nn.Module):
    def __init__(self, encoder_dim: int, encoder_rates: tuple[int, ...], latent_dim: int):
        super().__init__()
        layers: list[nn.Module] = [weight_norm(nn.Conv1d(1, encoder_dim, kernel_size=7, padding=3))]
        for stride in encoder_rates:
            encoder_dim *= 2
            layers.append(DAVEncoderBlock(encoder_dim, stride=stride))
        layers.extend(
            (
                MiniMaxMusic3Snake1d(encoder_dim),
                weight_norm(nn.Conv1d(encoder_dim, latent_dim, kernel_size=3, padding=1)),
            )
        )
        self.block = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states)


class DAVDecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            MiniMaxMusic3Snake1d(input_dim),
            weight_norm(
                nn.ConvTranspose1d(
                    input_dim,
                    output_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                )
            ),
            DAVResidualUnit(output_dim, dilation=1),
            DAVResidualUnit(output_dim, dilation=3),
            DAVResidualUnit(output_dim, dilation=9),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states)


class DAVDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, upsampling_ratios: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = [weight_norm(nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3))]
        output_dim = hidden_dim
        for index, stride in enumerate(upsampling_ratios):
            input_channels = hidden_dim // (2**index)
            output_dim = hidden_dim // (2 ** (index + 1))
            layers.append(DAVDecoderBlock(input_channels, output_dim, stride=stride))
        layers.extend(
            (
                MiniMaxMusic3Snake1d(output_dim),
                weight_norm(nn.Conv1d(output_dim, 1, kernel_size=7, padding=3)),
                nn.Tanh(),
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model(hidden_states)


class MiniMaxMusic3DAV(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        latent_channels: int = 128,
        channel_latent_channels: int = 64,
        encoder_dim: int = 64,
        encoder_rates: tuple[int, ...] = (2, 4, 8, 8),
        encoder_latent_dim: int = 1024,
        decoder_input_dim: int = 1024,
        decoder_hidden_dim: int = 1536,
        upsampling_ratios: tuple[int, ...] = (8, 8, 4, 2),
        sampling_rate: int = 44_100,
    ):
        super().__init__()
        if channel_latent_channels * 2 != latent_channels:
            raise ValueError("latent_channels must be twice channel_latent_channels")
        self.hop_length = math.prod(encoder_rates)
        self.encoder = DAVEncoder(encoder_dim, encoder_rates, encoder_latent_dim)
        self.mean_proj = nn.Conv1d(encoder_latent_dim, channel_latent_channels, kernel_size=1)
        self.logs_proj = nn.Conv1d(encoder_latent_dim, channel_latent_channels, kernel_size=1)
        self.dec_in_proj = nn.Conv1d(channel_latent_channels, decoder_input_dim, kernel_size=1)
        self.decoder = DAVDecoder(decoder_input_dim, decoder_hidden_dim, upsampling_ratios)

    def _prepare_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform[None, None]
        elif waveform.ndim == 2:
            waveform = waveform[None]
        if waveform.ndim != 3:
            raise ValueError("waveform must have shape [batch, channels, samples]")
        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)
        elif waveform.shape[1] != 2:
            raise ValueError("waveform must be mono or stereo")
        remainder = waveform.shape[-1] % self.hop_length
        if remainder:
            waveform = torch.nn.functional.pad(waveform, (0, self.hop_length - remainder))
        return waveform

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = self._prepare_waveform(waveform)
        batch_size = waveform.shape[0]
        hidden_states = self.encoder(waveform.reshape(batch_size * 2, 1, -1))
        latents = self.mean_proj(hidden_states)
        return latents.reshape(batch_size, self.config.latent_channels, -1)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 3 or latents.shape[1] != self.config.latent_channels:
            raise ValueError(f"latents must have shape [batch, {self.config.latent_channels}, frames]")
        batch_size, _, length = latents.shape
        hidden_states = latents.reshape(batch_size * 2, self.config.channel_latent_channels, length)
        waveform = self.decoder(self.dec_in_proj(hidden_states))
        return waveform.reshape(batch_size, 2, -1)

    forward = decode
