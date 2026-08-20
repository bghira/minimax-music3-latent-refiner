import unittest

import torch

from minimax_music3_latent_refiner.pipeline import (
    DAV_HOP,
    MiniMaxMusic3RefinerPipeline,
    prepare_waveform,
    window_starts,
)


class SelectionPipeline(MiniMaxMusic3RefinerPipeline):
    def __init__(self):
        self.audio_vae = object()
        self.selected_encoder = None

    def _refine_chunk(self, waveform, steps, audio_encoder):
        self.selected_encoder = audio_encoder
        return torch.zeros(waveform.shape[0], 128, 1)


class ExternalEncoder:
    def encode(self, waveform):
        return waveform


class PipelineUtilityTests(unittest.TestCase):
    def test_prepare_waveform_makes_stereo_batch(self):
        waveform = prepare_waveform(torch.zeros(100), 44_100)
        self.assertEqual((1, 2, 100), tuple(waveform.shape))

    def test_window_starts_cover_tail_on_dav_grid(self):
        starts = window_starts(100_000, 30_000, 2_000)
        self.assertEqual(0, starts[0])
        self.assertEqual(0, starts[-1] % DAV_HOP)
        self.assertGreaterEqual(starts[-1] + 30_000, 100_000 - DAV_HOP)

    def test_overlap_must_be_smaller_than_window(self):
        with self.assertRaisesRegex(ValueError, "smaller"):
            window_starts(100, 20, 20)

    def test_external_encoder_takes_precedence(self):
        pipeline = SelectionPipeline()
        external = ExternalEncoder()
        pipeline.refine_latents(torch.zeros(2, 100), 44_100, audio_encoder=external)
        self.assertIs(external, pipeline.selected_encoder)


if __name__ == "__main__":
    unittest.main()
