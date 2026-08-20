# ComfyUI node

Install from the directory containing ComfyUI:

```bash
git lfs install
git clone https://huggingface.co/terminusresearch/minimax-music3-latent-refiner-v0.10
ComfyUI/venv/bin/python -m pip install -e minimax-music3-latent-refiner-v0.10
ln -s "$PWD/minimax-music3-latent-refiner-v0.10/comfyui_node" \
  ComfyUI/custom_nodes/minimax_music3_latent_refiner
```

Restart ComfyUI. Load `comfyui_workflow_example.json` from the repository root.

`MiniMax Music3 Latent Refine` uses the encoder bundled with the refiner loader when `audio_vae_encoder` is not connected. A connected `AUDIO_VAE_ENCODER` always takes precedence. A stock MiniMax Music3 `VAE` may be connected to `audio_vae_decoder`; otherwise the bundled full DAV decodes the result.
