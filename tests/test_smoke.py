"""No-data smoke tests for the final MorphoMoE package.

Run with: ``python -m unittest tests.test_smoke``.
"""

import unittest

import torch

from models.morphomoe import MorphoMoE
from train.train import create_optimizer
from utils.losses import DiceLoss


class MorphoMoESmokeTest(unittest.TestCase):
    def test_public_model_import_and_training_components(self):
        model = MorphoMoE(in_channels=3, num_classes=4)
        optimizer = create_optimizer(
            model, {"optimizer": "adam", "learning_rate": 1e-4, "weight_decay": 1e-5}
        )
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertIsInstance(DiceLoss(), DiceLoss)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_forward_output_shape(self):
        model = MorphoMoE(in_channels=3, num_classes=4).cuda().eval()
        with torch.no_grad():
            logits = model(torch.randn(1, 3, 200, 200, device="cuda"))
        self.assertEqual(logits.shape, (1, 4, 200, 200))
