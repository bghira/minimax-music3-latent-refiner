# Copyright 2026 SimpleTuner contributors
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()
        inv_freq = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(length, device=device, dtype=torch.float32)
        angles = positions[:, None] * self.inv_freq.to(device)[None, :]
        return angles.cos(), angles.sin()


def apply_rope(states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first, second = states.chunk(2, dim=-1)
    cos = cos[None, None]
    sin = sin[None, None]
    return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)


class RefinerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, cond_dim: int):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.attn_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.adaln = nn.Linear(d_model, 6 * d_model)
        self.layer_cond_proj = nn.Linear(cond_dim, d_model, bias=False)

    def forward(
        self,
        states: torch.Tensor,
        conditioning: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer_conditioning: torch.Tensor,
    ) -> torch.Tensor:
        states = states + self.layer_cond_proj(layer_conditioning)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaln(conditioning).chunk(6, dim=-1)
        batch, length, _ = states.shape
        normed = self.attn_norm(states) * (1 + scale_a[:, None]) + shift_a[:, None]
        query, key, value = self.qkv(normed).view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query = apply_rope(query, cos, sin)
        key = apply_rope(key, cos, sin)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        states = states + gate_a[:, None] * self.attn_out(attended)
        normed = self.mlp_norm(states) * (1 + scale_m[:, None]) + shift_m[:, None]
        return states + gate_m[:, None] * self.mlp(normed)


class MiniMaxMusic3LatentRefiner(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        latent_dim: int = 128,
        cond_dim: int = 768,
        d_model: int = 768,
        depth: int = 12,
        heads: int = 12,
        mert_layer_count: int = 13,
        mert_input_layer: int = 7,
        style_dim: int = 512,
        task_count: int = 3,
    ):
        super().__init__()
        if not 0 <= mert_input_layer < mert_layer_count:
            raise ValueError("mert_input_layer must select an available MERT layer")
        self.proj_in = nn.Linear(latent_dim + cond_dim, d_model)
        self.time_dim = d_model
        self.time_embed = nn.Sequential(nn.Linear(256, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.style_proj = nn.Sequential(nn.Linear(style_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.style_null = nn.Parameter(torch.zeros(style_dim))
        self.task_embed = nn.Embedding(task_count + 1, d_model)
        self.context_embed = nn.Parameter(torch.zeros(d_model))
        self.degraded_in_proj = nn.Linear(latent_dim, d_model, bias=False)
        self.degraded_null = nn.Parameter(torch.zeros(latent_dim))
        self.mert_null = nn.Parameter(torch.zeros(mert_layer_count, cond_dim))
        self.rope = RotaryEmbedding(d_model // heads)
        self.blocks = nn.ModuleList(RefinerBlock(d_model, heads, cond_dim) for _ in range(depth))
        self.layer_map = [min(1 + index, mert_layer_count - 2) for index in range(depth)]
        self.out_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj_out = nn.Linear(d_model, latent_dim)

    @staticmethod
    def timestep_features(timestep: torch.Tensor) -> torch.Tensor:
        half = 128
        frequencies = torch.exp(
            -math.log(10_000.0) * torch.arange(half, device=timestep.device, dtype=torch.float32) / half
        )
        angles = timestep[:, None].float() * frequencies[None, :] * 1_000.0
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        conditioning: torch.Tensor,
        timestep: torch.Tensor,
        layer_conditioning: torch.Tensor,
        style: torch.Tensor,
        degraded_latents: torch.Tensor,
        context_latents: torch.Tensor,
        task: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_length = noisy_latents.shape[1]
        if conditioning.shape[1] != target_length or degraded_latents.shape[1] != target_length:
            raise ValueError("conditioning, degraded latents, and noisy latents must share a frame count")
        if context_latents.shape[1] != target_length:
            raise ValueError("context latents must share the target frame count")

        states = self.proj_in(torch.cat((noisy_latents, conditioning), dim=-1))
        states = states + self.degraded_in_proj(degraded_latents)
        context_states = self.proj_in(torch.cat((context_latents, conditioning), dim=-1))
        context_states = context_states + self.context_embed[None, None]
        states = torch.cat((context_states, states), dim=1)

        time_conditioning = self.time_embed(self.timestep_features(timestep))
        if task is None:
            task = torch.full(
                (timestep.shape[0],),
                self.task_embed.num_embeddings - 1,
                dtype=torch.long,
                device=timestep.device,
            )
        time_conditioning = time_conditioning + self.task_embed(task)
        time_conditioning = time_conditioning + self.style_proj(style)

        cos, sin = self.rope(target_length, states.device)
        cos = torch.cat((cos, cos), dim=0)
        sin = torch.cat((sin, sin), dim=0)
        for index, block in enumerate(self.blocks):
            block_layers = layer_conditioning[:, self.layer_map[index]]
            block_layers = torch.cat((block_layers, block_layers), dim=1)
            states = block(states, time_conditioning, cos, sin, block_layers)
        return self.proj_out(self.out_norm(states[:, target_length:]))


@torch.inference_mode()
def bridge_sample(
    model: MiniMaxMusic3LatentRefiner,
    conditioning: torch.Tensor,
    layer_conditioning: torch.Tensor,
    degraded_latents: torch.Tensor,
    style: torch.Tensor,
    steps: int = 32,
) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be positive")
    latents = degraded_latents.clone()
    schedule = torch.linspace(1.0, 0.0, steps + 1, device=latents.device)
    for index in range(steps):
        timestep = schedule[index].expand(latents.shape[0])
        velocity = model(
            latents,
            conditioning,
            timestep,
            layer_conditioning,
            style,
            degraded_latents,
            degraded_latents,
        )
        latents = latents - (schedule[index] - schedule[index + 1]) * velocity
    return latents
