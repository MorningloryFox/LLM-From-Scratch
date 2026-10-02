"""Train Feneco-Char locally and save a reproducible experiment record."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from llm_from_scratch.experiment import (
    append_jsonl,
    count_parameters,
    corpus_sha256,
    environment_summary,
    git_revision,
    git_working_tree_dirty,
    set_seed,
    timestamp_utc,
)
from llm_from_scratch.corpus import load_documents, split_documents
from llm_from_scratch.model import Feneco, ModelConfig, config_to_dict

console = Console()


def split_corpus(encoded: torch.Tensor, context_length: int) -> tuple[torch.Tensor, ...]:
    """Split contiguous character IDs into train, validation, and test segments."""
    train_end = int(0.8 * len(encoded))
    validation_end = int(0.9 * len(encoded))
    parts = (
        encoded[:train_end],
        encoded[train_end:validation_end],
        encoded[validation_end:],
    )
    if any(len(part) < context_length + 2 for part in parts):
        raise SystemExit(
            "Each 80/10/10 data split must contain at least context_length + 2 "
            f"characters ({context_length + 2}). Use a larger corpus or shorter context."
        )
    return parts


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(
        len(data) - context_length - 1, (batch_size,), generator=generator
    )
    x = torch.stack([data[start : start + context_length] for start in starts])
    y = torch.stack([data[start + 1 : start + context_length + 1] for start in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: Feneco,
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device,
    seed: int,
    batches: int,
    progress: Progress | None = None,
    description: str = "Avaliando",
) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    own_progress = progress is None
    if progress is None:
        progress = Progress(
            SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
            TaskProgressColumn(), TimeElapsedColumn(), console=console,
        )
        progress.start()
    task = progress.add_task(description, total=batches)
    try:
        for _ in range(batches):
            x, y = get_batch(data, batch_size, context_length, device, generator)
            _, loss = model(x, y)
            assert loss is not None
            losses.append(loss.item())
            progress.advance(task)
    finally:
        progress.remove_task(task)
        if own_progress:
            progress.stop()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True,
        help="UTF-8 .txt file or folder of .txt documents",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-batches", type=int, default=20)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/feneco-char-0.1.pt")
    )
    parser.add_argument("--metrics", type=Path, default=Path(".local/experiments.jsonl"))
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.evaluation_batches < 1:
        raise SystemExit("steps, batch-size, and evaluation-batches must be positive")
    documents = load_documents(args.data)
    text = "\n\n".join(content for _, content in documents)
    vocabulary = sorted(set(text))
    token_to_id = {character: index for index, character in enumerate(vocabulary)}
    encoded = torch.tensor([token_to_id[character] for character in text], dtype=torch.long)
    if args.data.is_dir():
        split_texts, split_files = split_documents(
            documents, args.context_length, args.seed
        )
        train_data, validation_data, test_data = (
            torch.tensor([token_to_id[character] for character in part], dtype=torch.long)
            for part in split_texts
        )
        split_strategy = "whole_documents"
    else:
        train_data, validation_data, test_data = split_corpus(
            encoded, args.context_length
        )
        split_files = ((), (), ())
        split_strategy = "contiguous_text_segments"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig(vocab_size=len(vocabulary), context_length=args.context_length)
    model = Feneco(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    counts = count_parameters(model)
    head_instances = config.number_of_layers * config.number_of_heads
    allowed_attention_pairs = (
        head_instances * config.context_length * (config.context_length + 1) // 2
    )
    train_generator = torch.Generator().manual_seed(args.seed + 1)

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="cyan", justify="right")
    summary.add_column()
    summary.add_row("Dispositivo", str(device))
    summary.add_row("Corpus", f"{len(encoded):,} caracteres • vocabulário {len(vocabulary)}")
    if args.data.is_dir():
        summary.add_row("Livros por divisão", f"{len(split_files[0])} treino • {len(split_files[1])} validação • {len(split_files[2])} teste")
    summary.add_row("Arquitetura", f"{config.number_of_layers} camadas • {config.number_of_heads} cabeças por camada")
    summary.add_row("Parâmetros", f"{counts['total_parameters']:,} no total • {counts['trainable_parameters']:,} treináveis")
    summary.add_row("Pares de atenção causal", f"{allowed_attention_pairs:,} no contexto completo")
    console.print(Panel(summary, title="[bold bright_cyan]Feneco-Char • treino[/]", border_style="cyan"))

    started = time.perf_counter()
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    recent_losses: list[float] = []
    progress = Progress(
        SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
        TaskProgressColumn(), TimeElapsedColumn(), console=console,
    )
    with progress:
        train_task = progress.add_task("Treinando", total=args.steps)
        for step in range(args.steps):
            model.train()
            x, y = get_batch(
                train_data, args.batch_size, args.context_length, device, train_generator
            )
            _, loss = model(x, y)
            assert loss is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            recent_losses.append(loss.item())
            progress.advance(train_task)

            if step == 0 or (step + 1) % 50 == 0 or step + 1 == args.steps:
                validation_loss = estimate_loss(
                    model, validation_data, args.batch_size, args.context_length,
                    device, args.seed + 2, args.evaluation_batches,
                    progress=progress, description="Validando",
                )
                train_loss = sum(recent_losses) / len(recent_losses)
                recent_losses.clear()
                progress.update(
                    train_task,
                    description=f"Treino • passo {step + 1}/{args.steps} • perda {train_loss:.4f} • validação {validation_loss:.4f}",
                )
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.state_dict().items()
                    }

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    test_loss = estimate_loss(
        model,
        test_data,
        args.batch_size,
        args.context_length,
        device,
        args.seed + 3,
        args.evaluation_batches,
    )
    duration_seconds = time.perf_counter() - started

    corpus_digest = corpus_sha256(args.data)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": config_to_dict(config),
            "model_state": model.cpu().state_dict(),
            "vocabulary": vocabulary,
            "training_steps": args.steps,
            "trainable_parameters": counts["trainable_parameters"],
            "seed": args.seed,
            "best_validation_loss": best_validation_loss,
            "test_loss": test_loss,
            "corpus_sha256": corpus_digest,
            "split_strategy": split_strategy,
            "split_files": {
                "train": list(split_files[0]),
                "validation": list(split_files[1]),
                "test": list(split_files[2]),
            },
        },
        args.checkpoint,
    )
    results = Table.grid(padding=(0, 2))
    results.add_column(style="cyan", justify="right")
    results.add_column()
    results.add_row("Melhor perda de validação", f"{best_validation_loss:.4f}")
    results.add_row("Perda no teste", f"{test_loss:.4f}")
    results.add_row("Checkpoint", str(args.checkpoint))
    console.print(Panel(results, title="[bold green]Treino concluído[/]", border_style="green"))

    record = {
        "timestamp_utc": timestamp_utc(),
        "model_name": "Feneco-Char-0.1",
        "code_revision": git_revision(),
        "code_working_tree_dirty": git_working_tree_dirty(),
        "corpus_path": str(args.data.resolve()),
        "corpus_sha256": corpus_digest,
        "corpus_characters": len(encoded),
        "vocabulary_size": len(vocabulary),
        "split_characters": {
            "train": len(train_data),
            "validation": len(validation_data),
            "test": len(test_data),
        },
        "split_strategy": split_strategy,
        "split_files": {
            "train": list(split_files[0]),
            "validation": list(split_files[1]),
            "test": list(split_files[2]),
        },
        "model_config": config_to_dict(config),
        **counts,
        "attention_head_instances": head_instances,
        "allowed_causal_attention_pairs_at_context": allowed_attention_pairs,
        "quantization": "none",
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "evaluation_batches": args.evaluation_batches,
        },
        "loss": {"best_validation": best_validation_loss, "held_out_test": test_loss},
        "duration_seconds": duration_seconds,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "environment": environment_summary(device),
    }
    append_jsonl(args.metrics, record)
    console.print(f"[dim]Métricas da execução salvas em {args.metrics}[/]")


if __name__ == "__main__":
    main()
