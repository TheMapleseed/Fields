import unittest

import torch

from farm_slm.device import (
    Accelerator,
    classify_cuda,
    classify_directml,
    detect_accelerator,
    select_accelerator,
)


class DeviceTests(unittest.TestCase):
    def test_nvidia_beats_amd_directml_and_metal(self) -> None:
        nvidia = Accelerator(torch.device("cuda"), "nvidia", "RTX")
        amd = Accelerator(torch.device("privateuseone", 0), "amd", "Radeon")
        metal = Accelerator(torch.device("mps"), "metal", "Apple GPU")
        chosen = select_accelerator(nvidia, amd, metal)
        self.assertEqual(chosen.vendor, "nvidia")

    def test_rocm_is_labeled_amd(self) -> None:
        self.assertEqual(classify_cuda(hip=True), "amd")
        self.assertEqual(classify_cuda(hip=False), "nvidia")

    def test_directml_name_picks_the_vendor(self) -> None:
        self.assertEqual(classify_directml("AMD Radeon RX 7900 XTX"), "amd")
        self.assertEqual(classify_directml("NVIDIA GeForce RTX 4090"), "nvidia")
        self.assertEqual(classify_directml("Intel Arc A770"), "intel")

    def test_metal_is_used_when_cuda_and_directml_are_absent(self) -> None:
        metal = Accelerator(torch.device("mps"), "metal", "Apple GPU")
        chosen = select_accelerator(None, None, metal)
        self.assertEqual(chosen.device.type, "mps")
        self.assertEqual(chosen.label(), "metal:Apple GPU")

    def test_cpu_is_the_fallback(self) -> None:
        chosen = select_accelerator(None, None, None)
        self.assertEqual(chosen.device.type, "cpu")

    def test_this_machine_returns_a_usable_device(self) -> None:
        found = detect_accelerator()
        self.assertIn(found.device.type, {"cpu", "cuda", "mps", "privateuseone"})
        self.assertTrue(found.vendor)
        self.assertIn(":", found.label())
