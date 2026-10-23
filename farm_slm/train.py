"""Train FarmSLM on collected text, or on a short primer, and write safetensors."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from farm_slm.config import ProjectConfig, load_project, project_root
from farm_slm.device import detect_accelerator
from farm_slm.locality import condition_text
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
    documents = _load_corpus(base, project.model.domains)
    source = "corpus" if documents else "primer"
    if not documents:
        missing = [domain for domain in project.model.domains if domain not in _PRIMERS]
        if missing:
            raise ValueError(f"no primer for {', '.join(missing)}")
        documents = [(domain, _PRIMERS[domain]) for domain in project.model.domains]
    token_ids = _fit_token_budget(documents, tokenizer, project.train.max_train_tokens)
    if len(token_ids) < 2:
        raise ValueError("not enough tokens to train")
    batch = torch.tensor([token_ids], dtype=torch.long, device=accelerator.device)
    losses = _optimize(model, batch, project)
    destination = base / project.weights_path
    model.save_safetensors(destination)
    print(f"device={accelerator.label()}")
    print(f"source={source}")
    for index, loss in enumerate(losses, start=1):
        print(f"step={index} loss={loss:.4f}")
    print(f"saved={destination.resolve()}")
    return destination


def _optimize(model: FarmSLM, batch: torch.Tensor, project: ProjectConfig) -> list[float]:
    settings = project.train
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    losses: list[float] = []
    for _ in range(settings.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch, labels=batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def _load_corpus(root: Path, domains: tuple[str, ...]) -> list[tuple[str, str]]:
    corpus = root / "data" / "corpus"
    if not corpus.is_dir():
        return []
    documents: list[tuple[str, str]] = []
    allowed = set(domains)
    for path in sorted(corpus.glob("*.jsonl")):
        if path.name.endswith(".partial"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            domain = row.get("domain")
            text = row.get("text")
            if domain not in allowed or not isinstance(text, str) or not text.strip():
                continue
            region = row.get("region", "unspecified")
            units = row.get("units", "unspecified")
            if not isinstance(region, str) or not isinstance(units, str):
                continue
            documents.append((str(domain), condition_text(text, region, units)))
    return documents


def _fit_token_budget(
    documents: list[tuple[str, str]],
    tokenizer: Tokenizer,
    limit: int,
) -> list[int]:
    packed: list[int] = []
    for domain, text in documents:
        if len(packed) >= limit:
            break
        piece = tokenizer.encode(text, domain=domain)
        room = limit - len(packed)
        packed.extend(piece[:room])
    return packed


def main() -> None:
    run_training()


if __name__ == "__main__":
    main()
