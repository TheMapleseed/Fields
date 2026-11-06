"""Save and resume training with safetensors for the model and the optimizer."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.optim import Optimizer

from farm_slm.model import FarmSLM

_MODEL_NAME = "model.safetensors"
_OPTIMIZER_NAME = "optimizer.safetensors"


class CheckpointError(ValueError):
    pass


def checkpoint_paths(directory: Path) -> tuple[Path, Path]:
    return directory / _MODEL_NAME, directory / _OPTIMIZER_NAME


def checkpoint_exists(directory: Path) -> bool:
    model_path, optimizer_path = checkpoint_paths(directory)
    return model_path.is_file() and optimizer_path.is_file()


def trainable_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def ensure_optimizer_state(optimizer: Optimizer, model: nn.Module) -> None:
    """Fill Adam slots for trainable weights that did not receive a gradient this step."""
    for _, param in trainable_parameters(model):
        state = optimizer.state.setdefault(param, {})
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(param)
        if "exp_avg_sq" not in state:
            state["exp_avg_sq"] = torch.zeros_like(param)
        if "step" not in state:
            state["step"] = torch.tensor(0, dtype=torch.int64)


def save_checkpoint(
    directory: Path,
    model: FarmSLM,
    optimizer: Optimizer,
    *,
    step: int,
    learning_rate: float,
    weight_decay: float,
) -> None:
    if step < 0:
        raise CheckpointError("checkpoint step cannot be negative")
    directory.mkdir(parents=True, exist_ok=True)
    model_path, optimizer_path = checkpoint_paths(directory)
    model.save_safetensors(model_path)
    tensors: dict[str, torch.Tensor] = {
        "step": torch.tensor([step], dtype=torch.int64),
    }
    named = trainable_parameters(model)
    if len(named) != len(optimizer.param_groups[0]["params"]):
        raise CheckpointError("optimizer parameter count does not match the model")
    for (name, param), opt_param in zip(named, optimizer.param_groups[0]["params"]):
        if param is not opt_param:
            raise CheckpointError(f"optimizer is out of order at {name}")
        state = optimizer.state.get(param)
        if state is None:
            raise CheckpointError(f"optimizer has no state for {name}")
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if not isinstance(exp_avg, torch.Tensor) or not isinstance(exp_avg_sq, torch.Tensor):
            raise CheckpointError(f"optimizer state for {name} is incomplete")
        safe = name.replace(".", "/")
        tensors[f"{safe}/exp_avg"] = exp_avg.detach().cpu().contiguous()
        tensors[f"{safe}/exp_avg_sq"] = exp_avg_sq.detach().cpu().contiguous()
        step_value = state.get("step")
        if isinstance(step_value, torch.Tensor):
            tensors[f"{safe}/adam_step"] = step_value.detach().cpu().contiguous()
        elif step_value is not None:
            tensors[f"{safe}/adam_step"] = torch.tensor([int(step_value)], dtype=torch.int64)
    temporary = optimizer_path.with_suffix(".safetensors.partial")
    try:
        save_file(
            tensors,
            str(temporary),
            metadata={
                "format": "farm_slm_optimizer",
                "step": str(step),
                "learning_rate": str(learning_rate),
                "weight_decay": str(weight_decay),
                "parameter_count": str(len(named)),
            },
        )
        temporary.replace(optimizer_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(
    directory: Path,
    model: FarmSLM,
    optimizer: Optimizer,
    *,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> int:
    model_path, optimizer_path = checkpoint_paths(directory)
    if not model_path.is_file() or not optimizer_path.is_file():
        raise CheckpointError("training checkpoint is incomplete")
    model.load_safetensors(model_path)
    model.to(device)
    with safe_open(str(optimizer_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = list(handle.keys())
        tensors = {key: handle.get_tensor(key) for key in keys}
    if metadata.get("format") != "farm_slm_optimizer":
        raise CheckpointError("optimizer file is not a FarmSLM checkpoint")
    if metadata.get("learning_rate") != str(learning_rate):
        raise CheckpointError("checkpoint learning rate does not match the config")
    if metadata.get("weight_decay") != str(weight_decay):
        raise CheckpointError("checkpoint weight decay does not match the config")
    if "step" not in tensors:
        raise CheckpointError("optimizer checkpoint is missing the training step")
    step = int(tensors["step"].reshape(-1)[0].item())
    named = trainable_parameters(model)
    if metadata.get("parameter_count") != str(len(named)):
        raise CheckpointError("checkpoint parameter count does not match the model")
    if len(named) != len(optimizer.param_groups[0]["params"]):
        raise CheckpointError("optimizer parameter count does not match the model")
    for (name, param), opt_param in zip(named, optimizer.param_groups[0]["params"]):
        if param is not opt_param:
            raise CheckpointError(f"optimizer is out of order at {name}")
        safe = name.replace(".", "/")
        exp_avg = tensors.get(f"{safe}/exp_avg")
        exp_avg_sq = tensors.get(f"{safe}/exp_avg_sq")
        if exp_avg is None or exp_avg_sq is None:
            raise CheckpointError(f"missing optimizer tensors for {name}")
        if tuple(exp_avg.shape) != tuple(param.shape) or tuple(exp_avg_sq.shape) != tuple(param.shape):
            raise CheckpointError(f"optimizer tensor shape mismatch for {name}")
        state = optimizer.state.setdefault(param, {})
        state["exp_avg"] = exp_avg.to(device=param.device, dtype=param.dtype)
        state["exp_avg_sq"] = exp_avg_sq.to(device=param.device, dtype=param.dtype)
        adam_step = tensors.get(f"{safe}/adam_step")
        if adam_step is None:
            state["step"] = torch.tensor(step, dtype=torch.int64)
        elif adam_step.ndim == 0:
            state["step"] = adam_step.to(dtype=torch.int64)
        else:
            state["step"] = adam_step.reshape(-1)[0].to(dtype=torch.int64)
    return step
