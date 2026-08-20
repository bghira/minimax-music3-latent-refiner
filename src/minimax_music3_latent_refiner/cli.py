from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

from .pipeline import DEFAULT_MODEL_ID, MiniMaxMusic3RefinerPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine audio with the MiniMax Music 3 bridge latent refiner")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--overlap-seconds", type=float, default=2.0)
    parser.add_argument("--direct", action="store_true", help="use one dense sequence instead of overlapping windows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, sample_rate = sf.read(args.input, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T.copy())
    pipeline = MiniMaxMusic3RefinerPipeline.from_pretrained(args.model, device=args.device)
    output = pipeline(
        waveform,
        sample_rate,
        steps=args.steps,
        window_seconds=None if args.direct else args.window_seconds,
        overlap_seconds=args.overlap_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, output.audio.squeeze(0).T.numpy(), output.sample_rate)


if __name__ == "__main__":
    main()
