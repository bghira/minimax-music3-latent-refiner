---
license: other
license_name: minimax-music3-community-license
license_link: https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE
library_name: diffusers
pipeline_tag: audio-to-audio
base_model: MiniMaxAI/MiniMax-Music3
tags:
- audio-restoration
- minimax-music3
- music
- diffusers
- comfyui
---

# MiniMax Music 3 Latent Refiner v0.10

This model takes damaged music and reconstructs a cleaner version while retaining the performance, timing, vocals, and arrangement. It operates in MiniMax Music 3's continuous DAV latent space. It is an audio refiner, not a text-to-music model and not an RVQ encoder.

The selected checkpoint restored an unseen holdout track substantially better than passing its damaged DAV latents directly through the decoder. The result is audible, not only a metric improvement.

## Quick start

```bash
git lfs install
git clone https://huggingface.co/terminusresearch/minimax-music3-latent-refiner-v0.10
cd minimax-music3-latent-refiner-v0.10
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

minimax-music3-refine input.wav refined.wav
```

Use current Diffusers main when working with the full MiniMax Music 3 pipeline:

```bash
python -m pip install --upgrade 'diffusers @ git+https://github.com/huggingface/diffusers.git'
```

CUDA is recommended. FP32 is the verified release precision.

## Python

```python
import soundfile as sf
import torch

from minimax_music3_latent_refiner import MiniMaxMusic3RefinerPipeline

audio, sample_rate = sf.read("input.wav", dtype="float32", always_2d=True)
refiner = MiniMaxMusic3RefinerPipeline.from_pretrained(
    "terminusresearch/minimax-music3-latent-refiner-v0.10",
    device="cuda",
)
result = refiner(torch.from_numpy(audio.T.copy()), sample_rate)
sf.write("refined.wav", result.audio.squeeze(0).T.numpy(), result.sample_rate)
```

The default path uses overlapping 30-second windows with a two-second latent overlap. Use `window_seconds=None` for one dense sequence. The checkpoint was trained on 30-second windows, so windowed inference is the quality baseline.

## Diffusers attachment

The helper attaches a `refine_audio` method to a loaded MiniMax Music 3 modular pipeline and uses its vocoder for decoding.

```python
from diffusers import ModularPipeline

from minimax_music3_latent_refiner import MiniMaxMusic3RefinerPipeline, attach_latent_refiner

music = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-Music3", trust_remote_code=True)
refiner = MiniMaxMusic3RefinerPipeline.from_pretrained(
    "terminusresearch/minimax-music3-latent-refiner-v0.10",
    device="cuda",
)
attach_latent_refiner(music, refiner)

result = music.refine_audio(waveform, sample_rate=44_100)
```

The attachment is instance-local. It does not modify the installed Diffusers package.

## ComfyUI

The custom node is in `comfyui_node/`. From the directory containing ComfyUI:

```bash
git lfs install
git clone https://huggingface.co/terminusresearch/minimax-music3-latent-refiner-v0.10
ComfyUI/venv/bin/python -m pip install -e minimax-music3-latent-refiner-v0.10
ln -s "$PWD/minimax-music3-latent-refiner-v0.10/comfyui_node" \
  ComfyUI/custom_nodes/minimax_music3_latent_refiner
```

Restart ComfyUI and load `comfyui_workflow_example.json`.

Nodes:

- `Load MiniMax Music3 Latent Refiner`
- `Load MiniMax Music3 Audio VAE Encoder`
- `MiniMax Music3 Latent Refine`

The refiner loader includes the full DAV encoder and decoder from `SimpleTuner/MiniMax-Music-3-Encoder`. `MiniMax Music3 Latent Refine` also exposes an optional `AUDIO_VAE_ENCODER` input. A connected encoder takes precedence. The optional `VAE` decoder input accepts ComfyUI's stock MiniMax Music 3 DAV.

## Selected checkpoint

| Item | Value |
|---|---:|
| Release | v0.10 |
| Parameters | 137,253,888 |
| Hybrid checkpoint | 1,000 steps |
| Total warm-start lineage | 2,000 steps |
| Training tracks | 2,262 |
| Holdout tracks | 32 |
| Training window | 30 seconds |
| Sampler | 32-step deterministic Euler bridge |
| Holdout diagonal cosine | 0.9479 |
| Holdout residual cosine | 0.7195 |

Checkpoint 2,000 reached 0.9405 diagonal cosine and 0.7105 residual cosine. Checkpoint 1,000 was selected.

`diagonal cosine` measures generated latent similarity to the matching clean target. `residual cosine` removes the damaged-input latent first, then measures whether the model's correction points toward the true clean correction. The second metric distinguishes refinement from passthrough.

## Runtime

Measured on one NVIDIA L40S with FP32 weights and 32 bridge steps:

| Audio | Refiner only | Speed | Refiner peak VRAM |
|---:|---:|---:|---:|
| 30 seconds | 2.67 seconds | 11.25x realtime | 0.92 GiB |
| 60 seconds | 8.02 seconds | 7.48x realtime | 1.30 GiB |
| 120 seconds | 27.68 seconds | 4.33x realtime | 2.06 GiB |

The complete 30-second path, including MERT, CLAP, DAV encode, bridge sampling, and DAV decode, took 4.26 seconds at 7.04x realtime with 7.15 GiB peak VRAM.

Long dense sequences fit because PyTorch SDPA uses a fused memory-efficient backend on the tested GPU. Compute remains quadratic. The release therefore defaults to overlapping windows.

## Architecture

The refiner receives the damaged audio through three simultaneous paths:

1. MERT hidden states provide frame-aligned musical features.
2. CLAP provides a pooled source-audio feature.
3. DAV latents enter both as an SR3-style per-frame stream and as in-context reference tokens.

The transformer sequence is:

```text
[damaged DAV reference tokens][bridge target tokens]
```

Both halves share frame positions. The sequence is bidirectional. The output head reads only the target half.

Bridge training interpolates between clean and damaged DAV latents:

```text
x(t) = (1 - t) * clean + t * damaged
target velocity = damaged - clean
```

Inference starts at the damaged endpoint and integrates from `t=1` to `t=0`.

## Experiment arc

The project began as a latent replanner for paired style-transfer audio. The early model combined per-layer MERT conditioning, pooled CLAP, and optional RVQ embeddings. Paired style-transfer data had weak frame correlation, so capacity and adapter variants could improve teacher-forced losses without producing reliable structure.

The restoration pivot made the target measurable: clean audio was corrupted at 44.1 kHz, and the model had to recover the original. Identity examples, explicit restore task conditioning, and residual-weighted metrics separated copying from correction.

The useful ablations were:

| Experiment | Result |
|---|---|
| Flow matching without a degraded stream | weak restoration |
| DDPM objective | residual cosine about 0.36 |
| Flow matching with SR3 degraded stream | residual cosine about 0.64 |
| Bridge transport | residual cosine 0.7015; diagonal cosine about 0.943 |
| Bridge + in-context reference + SR3 stream | residual cosine 0.7195; diagonal cosine 0.9479 |

Bridge transport supplied the correct endpoint geometry. The SR3 stream supplied aligned local evidence. In-context reference tokens allowed every target block to read the complete degraded latent sequence. Each addition produced a measured gain.

A separate on-policy flow-DPO branch was rejected. An unclamped preference loss destroyed sampler quality while ordinary training losses remained normal. Clamping reduced the damage but did not reverse it. Source-reject DPO remains an experiment and is not included in v0.10.

## Training data

The release used 2,294 music tracks split into 2,262 training tracks and 32 held-out tracks. Source and target were the same track. The source passed through a stochastic restoration degradation chain:

- bandwidth reduction
- additive noise
- bit-depth reduction
- soft clipping

The holdout degradation was deterministic. No training audio is distributed in this repository.

## Files

- `diffusion_pytorch_model.safetensors`: refiner weights
- `normalization.safetensors`: DAV and MERT normalization tensors
- `config.json`: refiner architecture
- `pipeline_config.json`: conditioning and sampling provenance
- `src/`: Apache-2.0 inference implementation
- `comfyui_node/`: ComfyUI integration

## Limitations

- Training windows were 30 seconds. Direct full-song attention was not trained or evaluated as the release path.
- This is a restoration model. It is not trained for source separation, remixing, cover generation, dereverberation as a distinct task, or arbitrary editing.
- MERT is required at runtime and is licensed CC-BY-NC-4.0.
- Exact output depends on the MiniMax Music 3 DAV encoder geometry and the provided normalization tensors.
- Strong corruption outside the training chain can remove information the model cannot reconstruct.

## Licenses

Repository code is Apache-2.0.

Use of the model depends on MiniMax Music 3 and is subject to the [MiniMax-Music3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE). Runtime conditioning uses [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M), licensed CC-BY-NC-4.0, and [larger_clap_music](https://huggingface.co/laion/larger_clap_music), licensed Apache-2.0.
