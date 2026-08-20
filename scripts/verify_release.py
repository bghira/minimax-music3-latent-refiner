from __future__ import annotations

import argparse
from pathlib import Path

from safetensors.torch import load_file

from minimax_music3_latent_refiner import MiniMaxMusic3LatentRefiner

EXPECTED_PARAMETERS = 137_253_888


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a MiniMax Music 3 latent refiner release directory")
    parser.add_argument("model", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = MiniMaxMusic3LatentRefiner.from_pretrained(args.model)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"expected {EXPECTED_PARAMETERS} parameters, found {parameters}")
    normalization = load_file(args.model / "normalization.safetensors")
    shapes = {key: tuple(value.shape) for key, value in normalization.items()}
    expected_shapes = {
        "latent_mean": (128,),
        "latent_std": (128,),
        "layer_mean": (13, 768),
        "layer_std": (13, 768),
    }
    if shapes != expected_shapes:
        raise RuntimeError(f"normalization shapes do not match: {shapes}")
    print(f"verified {parameters} parameters and normalization tensors")


if __name__ == "__main__":
    main()
