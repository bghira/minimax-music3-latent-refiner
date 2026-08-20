# AGENTS.md

## Project identity

- Repository: `minimax-music3-latent-refiner`
- Hub release: `terminusresearch/minimax-music3-latent-refiner-v0.10`
- Purpose: restore damaged audio in MiniMax Music 3 DAV latent space.
- Selected weights: `refiner3-bridge-hybrid/checkpoint-1000`.
- `v0.10` is the public release number for the selected bridge hybrid. It is not the later `replanner-8k-v10-incontext-dpo-src` experiment.
- The release model has 137,253,888 parameters. It was trained for 1,000 new steps after warm-starting the bridge checkpoint.

## Architecture invariants

- Audio rate: 44,100 Hz stereo.
- DAV hop: 512 waveform samples.
- DAV latent rate: `25 * 441 / 128 = 86.1328125` frames/second.
- DAV latents: `[batch, 128, frames]` externally and `[batch, frames, 128]` inside the refiner.
- MERT input: 24,000 Hz mono.
- MERT hidden states: 13 layers, 768 channels, linearly interpolated to DAV frame count.
- Main MERT conditioning layer: 7.
- Refiner: width 768, 12 blocks, 12 heads, head dimension 64.
- Block-to-MERT map: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11]`.
- CLAP conditioning comes from the degraded source audio, not the clean target.
- In-context layout is `[degraded reference tokens, bridge target tokens]`. Both halves use the same RoPE positions.
- The degraded latent stream is also injected per frame through `degraded_in_proj`.
- Restore sampling starts from degraded DAV latents at bridge time 1 and integrates to time 0.
- Normalization tensors are required. Never run the checkpoint without `normalization.safetensors`.

## Release behavior

- Default sampler: deterministic 32-step Euler bridge.
- Training windows were 30 seconds.
- Longer inputs must use overlapping windows by default. Direct long-sequence inference is supported by RoPE but is not the quality baseline.
- The package must preserve exact state-dict key compatibility with the SimpleTuner experiment checkpoint.
- An external ComfyUI `AUDIO_VAE_ENCODER` input takes precedence over the bundled encoder.
- ComfyUI's stock MiniMax Music 3 VAE is decoder-only. Do not call its `encode` method.

## Repository layout

- `src/minimax_music3_latent_refiner/model.py`: checkpoint-compatible refiner.
- `src/minimax_music3_latent_refiner/dav.py`: full DAV encoder/decoder.
- `src/minimax_music3_latent_refiner/pipeline.py`: conditioning, sampling, windowing, and audio I/O.
- `src/minimax_music3_latent_refiner/diffusers_patch.py`: opt-in pipeline attachment.
- `comfyui_node/`: ComfyUI custom node package.
- `examples/`: command-line and Diffusers examples.
- `scripts/`: release conversion and verification utilities.
- `tests/`: `unittest` tests.

## Development rules

- Use `.venv` and `python -m unittest -v`.
- Keep inference code independent of SimpleTuner imports.
- Do not copy DDP, DPO, dataset, or training-only code into this repository.
- Do not silently change tensor layout, frame rate, normalization, or conditioning source.
- Do not add fallback inference paths. Unsupported checkpoint or encoder formats must fail with a specific error.
- New behavior requires a focused unit test and one real-audio verification when it affects inference.
- Never commit or push unless the user explicitly requests it.

## Public text privacy

Never publish local machine identity in commits, model cards, Hub metadata, logs, examples, or validation notes.

Forbidden public text includes:

- Local absolute paths.
- Local account names or workstation usernames.
- Private pod paths.
- Raw terminal output containing local identity.
- Co-author trailers containing personal names or email addresses.

Use repository-relative paths and generic commands. Before any GitHub or Hugging Face publication, scan the exact public payload. If local identity is found, stop and report only:

`Blocked: local machine identity was found in public text.`
