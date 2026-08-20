import unittest

import torch

from minimax_music3_latent_refiner.model import MiniMaxMusic3LatentRefiner, bridge_sample


class ModelTests(unittest.TestCase):
    def make_model(self):
        return MiniMaxMusic3LatentRefiner(
            latent_dim=8,
            cond_dim=16,
            d_model=32,
            depth=2,
            heads=4,
            mert_layer_count=4,
            mert_input_layer=2,
            style_dim=12,
        ).eval()

    def test_forward_preserves_target_shape(self):
        model = self.make_model()
        noisy = torch.randn(1, 7, 8)
        layers = torch.randn(1, 4, 7, 16)
        output = model(
            noisy,
            layers[:, 2],
            torch.tensor([0.5]),
            layers,
            torch.randn(1, 12),
            torch.randn(1, 7, 8),
            torch.randn(1, 7, 8),
        )
        self.assertEqual((1, 7, 8), tuple(output.shape))

    def test_bridge_sampler_rejects_zero_steps(self):
        model = self.make_model()
        layers = torch.randn(1, 4, 7, 16)
        with self.assertRaisesRegex(ValueError, "positive"):
            bridge_sample(model, layers[:, 2], layers, torch.randn(1, 7, 8), torch.randn(1, 12), steps=0)


if __name__ == "__main__":
    unittest.main()
