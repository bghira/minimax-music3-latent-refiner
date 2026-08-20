from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a SimpleTuner latent refiner checkpoint for inference")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("normalization", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "model" not in checkpoint or "step" not in checkpoint:
        raise ValueError("checkpoint must contain model and step")
    normalization = torch.load(args.normalization, map_location="cpu", weights_only=True)
    required = {"latent_mean", "latent_std", "layer_mean", "layer_std"}
    if required.difference(normalization):
        raise ValueError("normalization checkpoint does not contain the required tensors")
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.contiguous() for key, value in checkpoint["model"].items()},
        args.output / "diffusion_pytorch_model.safetensors",
        metadata={"format": "pt", "step": str(checkpoint["step"])},
    )
    save_file(
        {key: normalization[key].contiguous() for key in sorted(required)},
        args.output / "normalization.safetensors",
        metadata={"format": "pt"},
    )


if __name__ == "__main__":
    main()
