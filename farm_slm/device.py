"""Choose the accelerator training should use on this machine.

NVIDIA CUDA wins, then AMD ROCm (PyTorch still calls that device CUDA), then
an installed DirectML adapter, then Apple Metal, then the CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Accelerator:
    device: torch.device
    vendor: str
    name: str

    def label(self) -> str:
        return f"{self.vendor}:{self.name}"


def select_accelerator(
    cuda: Accelerator | None,
    directml: Accelerator | None,
    metal: Accelerator | None,
) -> Accelerator:
    if cuda is not None:
        return cuda
    if directml is not None:
        return directml
    if metal is not None:
        return metal
    return Accelerator(torch.device("cpu"), "cpu", "CPU")


def classify_cuda(*, hip: bool) -> str:
    if hip:
        return "amd"
    return "nvidia"


def classify_directml(name: str) -> str:
    folded = name.casefold()
    if any(token in folded for token in ("nvidia", "geforce", "quadro", "rtx")):
        return "nvidia"
    if any(token in folded for token in ("amd", "radeon", "instinct")):
        return "amd"
    if "intel" in folded:
        return "intel"
    return "directml"


def detect_accelerator() -> Accelerator:
    return select_accelerator(_cuda_adapter(), _directml_adapter(), _metal_adapter())


def _cuda_adapter() -> Accelerator | None:
    if not torch.cuda.is_available():
        return None
    index = torch.cuda.current_device()
    hip = bool(getattr(torch.version, "hip", None))
    return Accelerator(
        torch.device("cuda", index),
        classify_cuda(hip=hip),
        torch.cuda.get_device_name(index),
    )


def _metal_adapter() -> Accelerator | None:
    backend = getattr(torch.backends, "mps", None)
    if backend is None or not backend.is_built() or not backend.is_available():
        return None
    return Accelerator(torch.device("mps"), "metal", "Apple GPU")


def _directml_adapter() -> Accelerator | None:
    try:
        import torch_directml
    except ImportError:
        return None
    count = int(torch_directml.device_count())
    if count < 1:
        return None
    name = str(torch_directml.device_name(0))
    return Accelerator(torch_directml.device(0), classify_directml(name), name)
