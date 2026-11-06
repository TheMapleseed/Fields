import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from farm_slm.checkpoint import (
    checkpoint_exists,
    ensure_optimizer_state,
    load_checkpoint,
    save_checkpoint,
    trainable_parameters,
)
from farm_slm.config import FarmSLMConfig, load_project
from farm_slm.model import FarmSLM


class CheckpointTests(unittest.TestCase):
    def test_optimizer_state_round_trips(self) -> None:
        project = load_project()
        config = FarmSLMConfig(
            vocab_size=project.model.vocab_size,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            segment_length=16,
            memory_slots=4,
            rope_theta=1_000_000.0,
            fope_floor_length=64,
            fope_init_gain=0.1,
            rms_norm_eps=1e-6,
            image_channels=3,
            patch_size=16,
            image_size=32,
            domains=project.model.domains,
        )
        torch.manual_seed(11)
        model = FarmSLM(config)
        trainable = [param for _, param in trainable_parameters(model)]
        optimizer = torch.optim.AdamW(trainable, lr=0.001, weight_decay=0.01)
        batch = torch.randint(0, config.vocab_size, (1, 8))
        loss = model(batch, labels=batch)
        loss.backward()
        optimizer.step()
        ensure_optimizer_state(optimizer, model)
        first = next(iter(trainable))
        before = optimizer.state[first]["exp_avg"].detach().clone()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            save_checkpoint(root, model, optimizer, step=3, learning_rate=0.001, weight_decay=0.01)
            self.assertTrue(checkpoint_exists(root))
            restored = FarmSLM(config)
            restored_opt = torch.optim.AdamW(
                [param for _, param in trainable_parameters(restored)],
                lr=0.001,
                weight_decay=0.01,
            )
            step = load_checkpoint(
                root,
                restored,
                restored_opt,
                learning_rate=0.001,
                weight_decay=0.01,
                device=torch.device("cpu"),
            )
            self.assertEqual(step, 3)
            again = next(iter(param for _, param in trainable_parameters(restored)))
            self.assertTrue(torch.allclose(restored_opt.state[again]["exp_avg"], before))
            self.assertTrue(
                torch.allclose(
                    restored.embed.weight.detach(),
                    model.embed.weight.detach(),
                )
            )
