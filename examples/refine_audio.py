import soundfile as sf
import torch

from minimax_music3_latent_refiner import MiniMaxMusic3RefinerPipeline

audio, sample_rate = sf.read("input.wav", dtype="float32", always_2d=True)
pipeline = MiniMaxMusic3RefinerPipeline.from_pretrained(
    "terminusresearch/minimax-music3-latent-refiner-v0.10",
    device="cuda",
)
result = pipeline(torch.from_numpy(audio.T.copy()), sample_rate)
sf.write("refined.wav", result.audio.squeeze(0).T.numpy(), result.sample_rate)
