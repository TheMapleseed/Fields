"""Train FarmSLM on ephemeral pulls, a local corpus, or a short primer.

Checkpoints save model + optimizer safetensors and a stage.kdl cursor so a
stopped run resumes the same catalog/PDF page automatically.
"""

from __future__ import annotations

from collections.abc import Callable
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
from farm_slm.ephemeral import (
    StepPuller,
    build_ephemeral_jobs,
    clean_ephemeral,
    make_step_puller,
)
from farm_slm.maps import read_chip
from farm_slm.model import FarmSLM
from farm_slm.stage import (
    StageError,
    StageState,
    jobs_fingerprint,
    load_stage,
    save_stage,
    stage_exists,
)
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
    documents, held, source, puller, jobs_sha = _training_material(base, project)
    finished = False
    try:
        checkpoint_dir = base / project.train.checkpoint_dir
        losses, started, finished = _optimize(
            model,
            documents,
            tokenizer,
            project,
            checkpoint_dir,
            accelerator.device,
            puller=puller,
            jobs_sha=jobs_sha,
        )
        destination = base / project.weights_path
        model.save_safetensors(destination)
        print(f"device={accelerator.label()}")
        print(f"source={source}")
        print(f"examples={len(documents) if puller is None else 'ephemeral-per-step'}")
        print(f"held_out={len(held)}")
        print(f"resume_step={started}")
        print(f"checkpoint={checkpoint_dir.resolve()}")
        if isinstance(puller, StepPuller):
            job = puller.cursor.current_job()
            print(
                "stage="
                f"job_index={puller.cursor.job_index} "
                f"page_index={puller.cursor.page_index} "
                f"job={job.title if job else 'none'}"
            )
        for index, loss in enumerate(losses, start=started + 1):
            print(f"step={index} loss={loss:.4f}")
        if held:
            hold_loss = _held_loss(model, held, tokenizer, project, accelerator.device)
            print(f"hold_out_loss={hold_loss:.4f}")
        print(f"saved={destination.resolve()}")
        return destination
    finally:
        # Keep the one-book PDF cache when the run stops early so resume is cheap.
        if project.collection.ephemeral_corpus and finished:
            clean_ephemeral(base)


def _training_material(
    base: Path,
    project: ProjectConfig,
) -> tuple[
    list[TrainingExample],
    list[TrainingExample],
    str,
    StepPuller | Callable[[int], TrainingExample | None] | None,
    str,
]:
    collection = project.collection
    if collection.fetch_during_train:
        jobs = build_ephemeral_jobs()
        puller = make_step_puller(
            jobs,
            project.model.domains,
            items_per_step=collection.ephemeral_items_per_step,
            root=base,
            settings=collection,
        )
        primers = _primer_examples(project.model.domains)
        return primers, [], "ephemeral", puller, jobs_fingerprint(jobs)

    documents, held = load_training_examples(
        base,
        project.model.domains,
        hold_out=project.train.hold_out_region,
    )
    if documents:
        return documents, held, "corpus", None, ""
    return _primer_examples(project.model.domains), held, "primer", None, ""


def _primer_examples(domains: tuple[str, ...]) -> list[TrainingExample]:
    missing = [domain for domain in domains if domain not in _PRIMERS]
    if missing:
        raise ValueError(f"no primer for {', '.join(missing)}")
    return [
        TrainingExample(domain, _PRIMERS[domain], "unspecified", "unspecified")
        for domain in domains
    ]


def _optimize(
    model: FarmSLM,
    documents: list[TrainingExample],
    tokenizer: Tokenizer,
    project: ProjectConfig,
    checkpoint_dir: Path,
    device: torch.device,
    *,
    puller: StepPuller | Callable[[int], TrainingExample | None] | None = None,
    jobs_sha: str = "",
) -> tuple[list[float], int, bool]:
    settings = project.train
    model.train()
    trainable = [param for _, param in trainable_parameters(model)]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    started = 0
    order = torch.randperm(len(documents)).tolist()
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
        restored = _restore_or_bootstrap_stage(
            checkpoint_dir,
            step=started,
            puller=puller,
            jobs_sha=jobs_sha,
            documents=documents,
            order=order,
        )
        if restored is not None:
            order = restored
        if started == settings.steps:
            return [], started, True
    losses: list[float] = []
    for step in range(started + 1, settings.steps + 1):
        example = _example_for_step(step, documents, order, puller)
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
            _save_training_stage(
                checkpoint_dir,
                step=step,
                puller=puller,
                jobs_sha=jobs_sha,
                order=tuple(order),
            )
    return losses, started, True


def _restore_or_bootstrap_stage(
    checkpoint_dir: Path,
    *,
    step: int,
    puller: StepPuller | Callable[[int], TrainingExample | None] | None,
    jobs_sha: str,
    documents: list[TrainingExample],
    order: list[int],
) -> list[int] | None:
    if stage_exists(checkpoint_dir):
        try:
            stage = load_stage(checkpoint_dir)
        except StageError as exc:
            print(f"stage_resume=skipped ({exc})")
            return None
        if jobs_sha and stage.jobs_sha not in {"", "none"} and stage.jobs_sha != jobs_sha:
            print(
                "stage_resume=reset (catalog fingerprint changed; keeping model weights)"
            )
            _save_training_stage(
                checkpoint_dir,
                step=step,
                puller=puller,
                jobs_sha=jobs_sha,
                order=tuple(order),
            )
            return None
        if isinstance(puller, StepPuller):
            puller.cursor.restore(stage.job_index, stage.page_index, stage.job_url)
            print(
                "stage_resume="
                f"job_index={stage.job_index} page_index={stage.page_index} "
                f"job={stage.job_title!r}"
            )
        if stage.order and len(stage.order) == len(documents):
            return list(stage.order)
        return None
    _save_training_stage(
        checkpoint_dir,
        step=step,
        puller=puller,
        jobs_sha=jobs_sha,
        order=tuple(order),
    )
    print("stage_resume=bootstrapped")
    return None


def _save_training_stage(
    checkpoint_dir: Path,
    *,
    step: int,
    puller: StepPuller | Callable[[int], TrainingExample | None] | None,
    jobs_sha: str,
    order: tuple[int, ...],
) -> None:
    job_index = 0
    page_index = 0
    job_url = ""
    job_title = ""
    if isinstance(puller, StepPuller):
        job_index, page_index, job_url, job_title = puller.cursor.snapshot()
    save_stage(
        checkpoint_dir,
        StageState(
            step=step,
            job_index=job_index,
            page_index=page_index,
            job_url=job_url,
            job_title=job_title,
            jobs_sha=jobs_sha or "none",
            order=order,
        ),
    )


def _example_for_step(
    step: int,
    documents: list[TrainingExample],
    order: list[int],
    puller: StepPuller | Callable[[int], TrainingExample | None] | None,
) -> TrainingExample:
    if puller is not None:
        pulled = puller(step)
        if pulled is not None:
            return pulled
    return documents[order[(step - 1) % len(order)]]


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
