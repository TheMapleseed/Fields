"""Project configuration loaded from farm_slm.kdl."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from farm_slm.kdl import KdlError, Node, parse_kdl

_DOMAIN_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_BYTE_VOCAB = 256
_SHARED_SPECIALS = 3  # pad, bos, eos


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FarmSLMConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    segment_length: int
    memory_slots: int
    rope_theta: float
    rms_norm_eps: float
    image_channels: int
    patch_size: int
    image_size: int
    domains: tuple[str, ...]

    def __post_init__(self) -> None:
        _positive("vocab_size", self.vocab_size)
        _positive("hidden_size", self.hidden_size)
        _positive("intermediate_size", self.intermediate_size)
        _positive("num_hidden_layers", self.num_hidden_layers)
        _positive("num_attention_heads", self.num_attention_heads)
        _positive("num_key_value_heads", self.num_key_value_heads)
        _positive("max_position_embeddings", self.max_position_embeddings)
        _positive("segment_length", self.segment_length)
        _positive("memory_slots", self.memory_slots)
        if self.rope_theta <= 0 or self.rms_norm_eps <= 0:
            raise ValueError("rope_theta and rms_norm_eps must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must divide evenly into attention heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("query heads must divide evenly into key/value heads")
        if self.segment_length > self.max_position_embeddings:
            raise ValueError("segment_length cannot exceed the context window")
        _positive("image_channels", self.image_channels)
        _positive("patch_size", self.patch_size)
        _positive("image_size", self.image_size)
        if self.image_channels > 4:
            raise ValueError("image_channels cannot exceed 4")
        if self.patch_size < 4 or self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size, and patch_size must be at least 4")
        if len(self.domains) != len(set(self.domains)):
            raise ValueError("domain names must be unique")
        for domain in self.domains:
            if not _DOMAIN_NAME.fullmatch(domain):
                raise ValueError(f"invalid domain name {domain!r}")
        minimum_vocab = _BYTE_VOCAB + _SHARED_SPECIALS + len(self.domains)
        if self.vocab_size < minimum_vocab:
            raise ValueError(f"vocab_size must be at least {minimum_vocab}")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def domain_token_id(self, domain: str) -> int:
        try:
            offset = self.domains.index(domain)
        except ValueError as exc:
            raise KeyError(domain) from exc
        return _BYTE_VOCAB + _SHARED_SPECIALS + offset


@dataclass(frozen=True)
class CollectionSettings:
    max_repos_per_query: int
    max_files_per_repo: int
    max_file_bytes: int
    max_total_bytes: int
    max_queries_per_domain: int

    def __post_init__(self) -> None:
        for name, value in (
            ("max_repos_per_query", self.max_repos_per_query),
            ("max_files_per_repo", self.max_files_per_repo),
            ("max_file_bytes", self.max_file_bytes),
            ("max_total_bytes", self.max_total_bytes),
            ("max_queries_per_domain", self.max_queries_per_domain),
        ):
            _positive(name, value)


@dataclass(frozen=True)
class TrainSettings:
    learning_rate: float
    steps: int
    weight_decay: float
    grad_clip: float
    seed: int
    max_train_tokens: int

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.grad_clip <= 0:
            raise ValueError("learning_rate and grad_clip must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        _positive("steps", self.steps)
        _positive("max_train_tokens", self.max_train_tokens)
        if self.seed < 0:
            raise ValueError("seed cannot be negative")


@dataclass(frozen=True)
class ProjectConfig:
    model: FarmSLMConfig
    collection: CollectionSettings
    train: TrainSettings
    weights_format: str
    weights_path: str

    def __post_init__(self) -> None:
        if self.weights_format != "safetensors":
            raise ValueError("weights format must be safetensors")
        if not self.weights_path.endswith(".safetensors"):
            raise ValueError("weights path must end in .safetensors")
        if Path(self.weights_path).is_absolute() or ".." in Path(self.weights_path).parts:
            raise ValueError("weights path must stay inside the project")


@dataclass(frozen=True)
class MapSettings:
    chip_size: int

    def __post_init__(self) -> None:
        if self.chip_size < 8 or self.chip_size > 2048 or self.chip_size % 2 != 0:
            raise ValueError("chip_size must be an even integer from 8 to 2048")


def load_map_settings(path: Path | None = None) -> MapSettings:
    nodes = _read_nodes(path)
    maps = _one(nodes, "maps")
    return MapSettings(chip_size=_as_int(_number(maps, "chip_size")))


def _read_nodes(path: Path | None) -> tuple[Node, ...]:
    config_path = project_root() / "farm_slm.kdl" if path is None else path
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read {config_path}") from exc
    try:
        return parse_kdl(text)
    except KdlError as exc:
        raise ValueError(f"invalid configuration: {exc}") from exc


def load_project(path: Path | None = None) -> ProjectConfig:
    nodes = _read_nodes(path)
    model_node = _one(nodes, "model")
    domain_node = _one(nodes, "domains")
    domains = tuple(_string_arg(child) for child in domain_node.children if child.name == "domain")
    if not domains:
        raise ValueError("domains must list at least one domain")
    model = FarmSLMConfig(
        vocab_size=_as_int(_number(model_node, "vocab_size")),
        hidden_size=_as_int(_number(model_node, "hidden_size")),
        intermediate_size=_as_int(_number(model_node, "intermediate_size")),
        num_hidden_layers=_as_int(_number(model_node, "num_hidden_layers")),
        num_attention_heads=_as_int(_number(model_node, "num_attention_heads")),
        num_key_value_heads=_as_int(_number(model_node, "num_key_value_heads")),
        max_position_embeddings=_as_int(_number(model_node, "max_position_embeddings")),
        segment_length=_as_int(_number(model_node, "segment_length")),
        memory_slots=_as_int(_number(model_node, "memory_slots")),
        rope_theta=float(_number(model_node, "rope_theta")),
        rms_norm_eps=float(_number(model_node, "rms_norm_eps")),
        image_channels=_as_int(_number(model_node, "image_channels")),
        patch_size=_as_int(_number(model_node, "patch_size")),
        image_size=_as_int(_number(model_node, "image_size")),
        domains=domains,
    )
    collection_node = _one(nodes, "collection")
    collection = CollectionSettings(
        max_repos_per_query=_as_int(_number(collection_node, "max_repos_per_query")),
        max_files_per_repo=_as_int(_number(collection_node, "max_files_per_repo")),
        max_file_bytes=_as_int(_number(collection_node, "max_file_bytes")),
        max_total_bytes=_as_int(_number(collection_node, "max_total_bytes")),
        max_queries_per_domain=_as_int(_number(collection_node, "max_queries_per_domain")),
    )
    train_node = _one(nodes, "train")
    train = TrainSettings(
        learning_rate=float(_number(train_node, "learning_rate")),
        steps=_as_int(_number(train_node, "steps")),
        weight_decay=float(_number(train_node, "weight_decay")),
        grad_clip=float(_number(train_node, "grad_clip")),
        seed=_as_int(_number(train_node, "seed")),
        max_train_tokens=_as_int(_number(train_node, "max_train_tokens")),
    )
    weights_node = _one(nodes, "weights")
    return ProjectConfig(
        model=model,
        collection=collection,
        train=train,
        weights_format=_string_arg(_child(weights_node, "format")),
        weights_path=_string_arg(_child(weights_node, "path")),
    )


def _one(nodes: tuple[Node, ...], name: str) -> Node:
    matches = [node for node in nodes if node.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one {name} node")
    return matches[0]


def _child(node: Node, name: str) -> Node:
    matches = [child for child in node.children if child.name == name]
    if len(matches) != 1:
        raise ValueError(f"{node.name} is missing {name}")
    return matches[0]


def _number(node: Node, name: str) -> int | float:
    child = _child(node, name)
    if len(child.args) != 1 or not isinstance(child.args[0], (int, float)):
        raise ValueError(f"{name} must be one number")
    return child.args[0]


def _string_arg(node: Node) -> str:
    if len(node.args) != 1 or not isinstance(node.args[0], str) or not node.args[0]:
        raise ValueError(f"{node.name} must be one non-empty string")
    return node.args[0]


def _as_int(value: int | float) -> int:
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("expected an integer")
    return int(value)


def _positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
