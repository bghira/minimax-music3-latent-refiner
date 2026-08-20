import torch
from diffusers import ModularPipeline

from minimax_music3_latent_refiner import MiniMaxMusic3RefinerPipeline, attach_latent_refiner

music = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-Music3", trust_remote_code=True)
refiner = MiniMaxMusic3RefinerPipeline.from_pretrained(
    "terminusresearch/minimax-music3-latent-refiner-v0.10",
    device="cuda",
)
attach_latent_refiner(music, refiner)

waveform = torch.zeros(1, 2, 44_100 * 10)
result = music.refine_audio(waveform, 44_100)
