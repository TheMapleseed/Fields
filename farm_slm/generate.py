"""Continue a prompt with a loaded FarmSLM checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from farm_slm.config import load_project, project_root
from farm_slm.device import detect_accelerator
from farm_slm.locality import condition_text
from farm_slm.maps import read_chip
from farm_slm.model import FarmSLM
from farm_slm.tokenizer import EOS_ID, Tokenizer


def generate(
    prompt: str,
    domain: str,
    *,
    root: Path | None = None,
    region: str = "unspecified",
    units: str = "unspecified",
    chip: Path | None = None,
    max_new_tokens: int = 128,
) -> str:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    base = project_root() if root is None else root
    project = load_project(base / "farm_slm.kdl")
    if domain not in project.model.domains:
        raise ValueError(f"unknown domain {domain!r}")
    accelerator = detect_accelerator()
    model = FarmSLM(project.model).to(accelerator.device)
    weights = base / project.weights_path
    if not weights.is_file():
        raise FileNotFoundError(f"missing weights at {weights}")
    model.load_safetensors(weights)
    model.eval()
    tokenizer = Tokenizer(project.model)
    text = condition_text(prompt, region, units)
    token_ids = tokenizer.encode_prompt(text, domain=domain)
    images = None
    if chip is not None:
        images = read_chip(chip).unsqueeze(0).to(device=accelerator.device)
    generated: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            batch = torch.tensor([token_ids + generated], dtype=torch.long, device=accelerator.device)
            if batch.shape[1] >= project.model.max_position_embeddings:
                break
            logits = model(batch, images=images)
            next_id = int(torch.argmax(logits[0, -1]).item())
            if next_id == EOS_ID:
                break
            generated.append(next_id)
    return tokenizer.decode_text(generated)


def main() -> None:
    text = generate(
        "Soil chemistry covers pH, cation exchange, nitrogen,",
        "chemistry",
        max_new_tokens=32,
    )
    print(text)


if __name__ == "__main__":
    main()
