"""Train FarmSLM on collected text, or on a short primer, and write safetensors."""

from __future__ import annotations

from pathlib import Path

import torch

from farm_slm.checkpoint import (
    CheckpointError,
    checkpoint_exists,
    ensure_optimizer_state,
    load_checkpoint,
    save_checkpoint,
    trainable_parameters,
)
from farm_slm.config import ProjectConfig, load_project, project_root
from farm_slm.corpus import TrainingExample, load_training_examples
from farm_slm.device import detect_accelerator
from farm_slm.maps import read_chip
from farm_slm.model import FarmSLM
from farm_slm.tokenizer import Tokenizer

_PRIMERS: dict[str, str] = {
    "english": (
        "Write plain English for a farmer. Prefer short sentences. "
        "Define a technical word the first time you use it."
    ),
    "agricultural_linguistics": (
        "Field words include tilth, loam, headland, volunteer crop, lodging, and scouting. "
        "Use those words with their farm meanings."
    ),
    "chemistry": (
        "Soil chemistry covers pH, cation exchange, nitrogen, phosphorus, potassium, "
        "and organic matter."
    ),
    "weather_reading": (
        "A farm forecast names temperature, dew point, wind, pressure, precipitation, "
        "and frost risk."
    ),
    "physics": (
        "Farm physics covers evapotranspiration, hydraulic head, soil moisture tension, "
        "and solar radiation."
    ),
}


def run_training(root: Path | None = None) -> Path:
    base = project_root() if root is None else root
    project = load_project(base / "farm_slm.kdl")
    accelerator = detect_accelerator()
    torch.manual_seed(project.train.seed)
    if accelerator.device.type == "cuda":
        torch.cuda.manual_seed_all(project.train.seed)
    model = FarmSLM(project.model).to(accelerator.device)
    tokenizer = Tokenizer(project.model)
    documents, held = load_training_examples(
        base,
        project.model.domains,
        hold_out=project.train.hold_out_region,
    )
    source = "corpus" if documents else "primer"
    if not documents:
        missing = [domain for domain in project.model.domains if domain not in _PRIMERS]
        if missing:
            raise ValueError(f"no primer for {', '.join(missing)}")
        documents = [
            TrainingExample(domain, _PRIMERS[domain], "unspecified", "unspecified")
            for domain in project.model.domains
        ]
    checkpoint_dir = base / project.train.checkpoint_dir
    losses, started = _optimize(
        model,
        documents,
        tokenizer,
        project,
        checkpoint_dir,
        accelerator.device,
    )
    destination = base / project.weights_path
    model.save_safetensors(destination)
    print(f"device={accelerator.label()}")
    print(f"source={source}")
    print(f"examples={len(documents)}")
    print(f"held_out={len(held)}")
    print(f"resume_step={started}")
    print(f"checkpoint={checkpoint_dir.resolve()}")
    for index, loss in enumerate(losses, start=started + 1):
        print(f"step={index} loss={loss:.4f}")
    if held:
        hold_loss = _held_loss(model, held, tokenizer, project, accelerator.device)
        print(f"hold_out_loss={hold_loss:.4f}")
    print(f"saved={destination.resolve()}")
    return destination


def _optimize(
    model: FarmSLM,
    documents: list[TrainingExample],
    tokenizer: Tokenizer,
    project: ProjectConfig,
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[list[float], int]:
    settings = project.train
    model.train()
    trainable = [param for _, param in trainable_parameters(model)]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    started = 0
    if settings.resume and checkpoint_exists(checkpoint_dir):
        started = load_checkpoint(
            checkpoint_dir,
            model,
            optimizer,
            learning_rate=settings.learning_rate,
            weight_decay=settings.weight_decay,
            device=device,
        )
        if started > settings.steps:
            raise CheckpointError(
                f"checkpoint step {started} is past train.steps {settings.steps}"
            )
        if started == settings.steps:
            return [], started
    order = torch.randperm(len(documents)).tolist()
    losses: list[float] = []
    for step in range(started + 1, settings.steps + 1):
        example = documents[order[(step - 1) % len(order)]]
        batch, images = _batch_example(example, tokenizer, settings.max_train_tokens, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch, labels=batch, images=images)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, settings.grad_clip)
        optimizer.step()
        ensure_optimizer_state(optimizer, model)
        losses.append(float(loss.detach()))
        if step % settings.checkpoint_every == 0 or step == settings.steps:
            save_checkpoint(
                checkpoint_dir,
                model,
                optimizer,
                step=step,
                learning_rate=settings.learning_rate,
                weight_decay=settings.weight_decay,
            )
    return losses, started


def _held_loss(
    model: FarmSLM,
    held: list[TrainingExample],
    tokenizer: Tokenizer,
    project: ProjectConfig,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for example in held:
            batch, images = _batch_example(
                example,
                tokenizer,
                project.train.max_train_tokens,
                device,
            )
            total += float(model(batch, labels=batch, images=images).detach())
    model.train()
    return total / len(held)


def _batch_example(
    example: TrainingExample,
    tokenizer: Tokenizer,
    limit: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    token_ids = tokenizer.encode(example.text, domain=example.domain)[:limit]
    if len(token_ids) < 2:
        raise ValueError("not enough tokens to train")
    batch = torch.tensor([token_ids], dtype=torch.long, device=device)
    images = None
    if example.chip is not None:
        images = read_chip(example.chip).unsqueeze(0).to(device=device)
    return batch, images


def main() -> None:
    run_training()


if __name__ == "__main__":
    main()
