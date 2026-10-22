"""FarmSLM decoder.

The context window is the configured maximum, 500000 tokens in farm_slm.kdl.
Full attention over that span is not how this model spends memory. The forward
pass splits the document into segments. Each layer keeps a fixed bank of memory
slots, rewrites those slots from the current segment, and passes them to the
next segment. Positions used by rotary embeddings are absolute indexes in the
full document.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from farm_slm.config import FarmSLMConfig

_CONTEXT_METADATA = "context_window"


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).type_as(x)


def memory_causal_mask(tokens: int, memory: int, device: torch.device) -> torch.Tensor:
    """True where a token may look. Memory is past context. The segment is causal."""
    allowed = torch.zeros(tokens, memory + tokens, dtype=torch.bool, device=device)
    allowed[:, :memory] = True
    allowed[:, memory:] = torch.tril(torch.ones(tokens, tokens, dtype=torch.bool, device=device))
    return allowed


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = query.shape[-1]
    half = head_dim // 2
    exponents = torch.arange(0, half, device=query.device, dtype=torch.float32) / half
    inv_freq = 1.0 / (theta ** exponents)
    angles = torch.outer(positions.to(dtype=torch.float32), inv_freq)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)[None, None, :, :]
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)[None, None, :, :]

    def twist(tensor: torch.Tensor) -> torch.Tensor:
        values = tensor.float()
        first, second = values[..., :half], values[..., half:]
        rotated = torch.cat((-second, first), dim=-1)
        return values * cos + rotated * sin

    return twist(query).type_as(query), twist(key).type_as(key)


def repeat_kv(tensor: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return tensor
    return tensor.repeat_interleave(repeats, dim=1)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: FarmSLMConfig) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.repeats = self.heads // self.kv_heads
        self.theta = config.rope_theta
        hidden = config.hidden_size
        self.q = nn.Linear(hidden, self.heads * self.head_dim, bias=False)
        self.k = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(self.heads * self.head_dim, hidden, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, _ = hidden.shape
        memory_slots = memory.shape[1]
        query = _heads(self.q(hidden), batch, tokens, self.heads, self.head_dim)
        key = _heads(self.k(hidden), batch, tokens, self.kv_heads, self.head_dim)
        value = _heads(self.v(hidden), batch, tokens, self.kv_heads, self.head_dim)
        query, key = apply_rope(query, key, positions, self.theta)
        memory_key = _heads(self.k(memory), batch, memory_slots, self.kv_heads, self.head_dim)
        memory_value = _heads(self.v(memory), batch, memory_slots, self.kv_heads, self.head_dim)
        key = torch.cat((memory_key, key), dim=2)
        value = torch.cat((memory_value, value), dim=2)
        key = repeat_kv(key, self.repeats)
        value = repeat_kv(value, self.repeats)
        scale = self.head_dim ** -0.5
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
        mask = memory_causal_mask(tokens, memory_slots, hidden.device)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).type_as(value)
        mixed = torch.matmul(weights, value)
        mixed = mixed.transpose(1, 2).contiguous().view(batch, tokens, -1)
        return self.o(mixed)


class MemoryWriter(nn.Module):
    """Rewrite the fixed memory slots from a finished segment."""

    def __init__(self, config: FarmSLMConfig) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.repeats = self.heads // self.kv_heads
        hidden = config.hidden_size
        self.q = nn.Linear(hidden, self.heads * self.head_dim, bias=False)
        self.k = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(self.heads * self.head_dim, hidden, bias=False)

    def forward(self, segment: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch, slots, _ = memory.shape
        tokens = segment.shape[1]
        if tokens == 0:
            return torch.zeros_like(memory)
        query = _heads(self.q(memory), batch, slots, self.heads, self.head_dim)
        key = _heads(self.k(segment), batch, tokens, self.kv_heads, self.head_dim)
        value = _heads(self.v(segment), batch, tokens, self.kv_heads, self.head_dim)
        key = repeat_kv(key, self.repeats)
        value = repeat_kv(value, self.repeats)
        scale = self.head_dim ** -0.5
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1).type_as(value)
        mixed = torch.matmul(weights, value)
        mixed = mixed.transpose(1, 2).contiguous().view(batch, slots, -1)
        return self.o(mixed)


class SwiGLU(nn.Module):
    def __init__(self, config: FarmSLMConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        inner = config.intermediate_size
        self.w1 = nn.Linear(hidden, inner, bias=False)
        self.w2 = nn.Linear(inner, hidden, bias=False)
        self.w3 = nn.Linear(hidden, inner, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(hidden)) * self.w3(hidden))


class Block(nn.Module):
    def __init__(self, config: FarmSLMConfig) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.norm_mlp = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)
        self.norm_segment = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm_memory = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.writer = MemoryWriter(config)

    def forward(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = hidden + self.attention(self.norm_attn(hidden), memory, positions)
        hidden = hidden + self.mlp(self.norm_mlp(hidden))
        updated = memory + self.writer(self.norm_segment(hidden), self.norm_memory(memory))
        return hidden, updated


class FarmSLM(nn.Module):
    def __init__(self, config: FarmSLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.memory_init = nn.Parameter(
            torch.empty(config.num_hidden_layers, config.memory_slots, config.hidden_size)
        )
        self.lm_head.weight = self.embed.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.memory_init, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must match input_ids")
        batch, length = input_ids.shape
        if length == 0:
            raise ValueError("input_ids is empty")
        if length > self.config.max_position_embeddings:
            raise ValueError(
                f"context length {length} exceeds {self.config.max_position_embeddings}"
            )
        if labels is not None and length < 2:
            raise ValueError("next-token loss needs at least two tokens")

        hidden_states = self.embed(input_ids)
        memories = [
            self.memory_init[layer].unsqueeze(0).expand(batch, -1, -1).clone()
            for layer in range(len(self.blocks))
        ]
        losses: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        logit_chunks: list[torch.Tensor] = []
        segment = self.config.segment_length
        targets = input_ids if labels is None else labels

        for start in range(0, length, segment):
            end = min(start + segment, length)
            hidden = hidden_states[:, start:end]
            positions = torch.arange(start, end, device=input_ids.device)
            for index, block in enumerate(self.blocks):
                hidden, memories[index] = block(hidden, memories[index], positions)
            hidden = self.norm(hidden)
            logits = self.lm_head(hidden)
            if labels is None:
                logit_chunks.append(logits)
                continue
            if end < length:
                target = targets[:, start + 1 : end + 1]
                prediction = logits
            else:
                target = targets[:, start + 1 : end]
                prediction = logits[:, : target.shape[1], :]
            if target.shape[1] == 0:
                continue
            losses.append(
                F.cross_entropy(
                    prediction.reshape(-1, self.config.vocab_size),
                    target.reshape(-1),
                    ignore_index=-100,
                )
            )
            weights.append(prediction.new_tensor(target.shape[1]))

        if labels is None:
            return torch.cat(logit_chunks, dim=1)
        if not losses:
            raise ValueError("no tokens contributed to the loss")
        total = torch.stack(weights).sum()
        weighted = torch.stack([loss * weight for loss, weight in zip(losses, weights)])
        return weighted.sum() / total

    def export_state_dict(self) -> dict[str, torch.Tensor]:
        state = self.state_dict()
        state.pop("lm_head.weight")
        return {key: value.detach().cpu().contiguous() for key, value in state.items()}

    def save_safetensors(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        domains = ",".join(self.config.domains)
        save_file(
            self.export_state_dict(),
            str(path),
            metadata={
                "format": "farm_slm",
                "weight_format": "safetensors",
                _CONTEXT_METADATA: str(self.config.max_position_embeddings),
                "segment_length": str(self.config.segment_length),
                "memory_slots": str(self.config.memory_slots),
                "domains": domains,
            },
        )

    def load_safetensors(self, path: Path) -> None:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            state = {key: handle.get_tensor(key) for key in handle.keys()}
        expected = str(self.config.max_position_embeddings)
        if metadata.get(_CONTEXT_METADATA) != expected:
            raise ValueError("safetensors file does not match this context window")
        if metadata.get("domains") != ",".join(self.config.domains):
            raise ValueError("safetensors file does not match these domains")
        incompatible = self.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(f"unexpected weights: {incompatible.unexpected_keys}")
        if set(incompatible.missing_keys) != {"lm_head.weight"}:
            raise ValueError(f"missing weights: {incompatible.missing_keys}")
        self.lm_head.weight = self.embed.weight


def architecture_report(model: FarmSLM) -> str:
    config = model.config
    params = sum(parameter.numel() for parameter in model.parameters())
    memory_bytes = config.num_hidden_layers * config.memory_slots * config.hidden_size * 2
    segment_scores = (
        config.segment_length
        * (config.segment_length + config.memory_slots)
        * config.num_attention_heads
        * 2
    )
    lines = [
        "FarmSLM",
        f"context_window={config.max_position_embeddings}",
        f"segment_length={config.segment_length}",
        f"memory_slots_per_layer={config.memory_slots}",
        f"layers={config.num_hidden_layers}",
        f"hidden={config.hidden_size}",
        f"query_heads={config.num_attention_heads}",
        f"kv_heads={config.num_key_value_heads}",
        f"head_dim={config.head_dim}",
        f"intermediate={config.intermediate_size}",
        f"vocab={config.vocab_size}",
        f"rope_theta={config.rope_theta}",
        f"parameters={params}",
        f"carried_memory_fp16_bytes={memory_bytes}",
        f"segment_attention_scores_fp16_bytes={segment_scores}",
        "domains=" + ",".join(config.domains),
        "weights=safetensors",
    ]
    return "\n".join(lines)


def _heads(
    tensor: torch.Tensor,
    batch: int,
    time: int,
    heads: int,
    head_dim: int,
) -> torch.Tensor:
    return tensor.view(batch, time, heads, head_dim).transpose(1, 2).contiguous()
