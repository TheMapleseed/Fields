import unittest

import torch

from farm_slm.config import FarmSLMConfig, load_project
from farm_slm.model import FarmSLM, architecture_report, memory_causal_mask
from farm_slm.tokenizer import Tokenizer
from farm_slm.train import run_training


class ModelTests(unittest.TestCase):
    def test_mask_keeps_memory_and_blocks_future_tokens(self) -> None:
        mask = memory_causal_mask(4, 3, torch.device("cpu"))
        self.assertEqual(mask[0, :3].tolist(), [True, True, True])
        self.assertEqual(mask[0, 3:].tolist(), [True, False, False, False])
        self.assertTrue(bool(mask[3].all()))

    def test_tokenizer_roundtrip_and_domain_token(self) -> None:
        config = load_project().model
        tokenizer = Tokenizer(config)
        text = "pH 6.5, 20°C"
        token_ids = tokenizer.encode(text, "chemistry")
        self.assertEqual(token_ids[0], tokenizer.bos_id)
        self.assertEqual(token_ids[1], config.domain_token_id("chemistry"))
        self.assertEqual(token_ids[-1], tokenizer.eos_id)
        self.assertEqual(tokenizer.decode_text(token_ids), text)

    def test_segment_memory_learns_and_rejects_overlong_context(self) -> None:
        project = load_project()
        domains = project.model.domains
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
            rms_norm_eps=1e-6,
            domains=domains,
        )
        torch.manual_seed(7)
        model = FarmSLM(config)
        batch = torch.randint(0, config.vocab_size, (2, 40))
        logits = model(batch)
        self.assertEqual(tuple(logits.shape), (2, 40, config.vocab_size))
        loss = model(batch, labels=batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.memory_init.grad)
        self.assertIsNotNone(model.embed.weight.grad)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 65, dtype=torch.long))

    def test_training_writes_safetensors(self) -> None:
        saved = run_training()
        self.assertTrue(saved.is_file())
        model = FarmSLM(load_project().model)
        model.load_safetensors(saved)
        self.assertEqual(model.lm_head.weight.data_ptr(), model.embed.weight.data_ptr())
        self.assertIn("context_window=500000", architecture_report(model))


if __name__ == "__main__":
    unittest.main()
