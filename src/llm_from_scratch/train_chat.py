"""Locally fine-tune a character Transformer on assistant turns in chat JSONL."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from torch.nn import functional as F

from llm_from_scratch.experiment import append_jsonl, count_parameters, environment_summary, set_seed, timestamp_utc
from llm_from_scratch.model import Feneco, ModelConfig, config_to_dict

console = Console()
END_ASSISTANT = "¤"
UNKNOWN = "�"
ROLE_PREFIX = {"system": "### Sistema:\n", "user": "### Usuário:\n", "assistant": "### Assistente:\n"}


def load_records(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"JSON inválido em {path}:{line_number}: {error}") from error
    return rows


def serialize_record(record: dict[str, Any]) -> str:
    result = []
    for message in record.get("messages", []):
        prefix = ROLE_PREFIX.get(str(message.get("role", "")).lower())
        content = str(message.get("content", "")).strip()
        if not prefix or not content:
            continue
        result.append(prefix + content + (END_ASSISTANT if prefix == ROLE_PREFIX["assistant"] else "") + "\n")
    return "".join(result)


def make_windows(
    sequences: list[list[int]], targets: list[list[int]], context: int
) -> list[tuple[int, int]]:
    windows = []
    for sequence_index, labels in enumerate(targets):
        for start in range(max(0, len(labels) - context + 1)):
            if any(label != -100 for label in labels[start : start + context]):
                windows.append((sequence_index, start))
    if not windows:
        raise SystemExit("Não há respostas do assistente longas o suficiente para formar lotes.")
    return windows


def get_batch(
    sequences: list[list[int]], targets: list[list[int]], windows: list[tuple[int, int]],
    batch_size: int, context: int, device: torch.device, generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    choices = torch.randint(len(windows), (batch_size,), generator=generator).tolist()
    x_rows, y_rows = [], []
    for choice in choices:
        sequence_index, start = windows[choice]
        x_rows.append(sequences[sequence_index][start : start + context])
        y_rows.append(targets[sequence_index][start : start + context])
    return torch.tensor(x_rows, dtype=torch.long, device=device), torch.tensor(y_rows, dtype=torch.long, device=device)


@torch.no_grad()
def estimate_loss(
    model: Feneco, sequences: list[list[int]], targets: list[list[int]],
    windows: list[tuple[int, int]], batch_size: int, context: int,
    device: torch.device, generator: torch.Generator, batches: int,
    progress: Progress | None = None, description: str = "Validando",
) -> float:
    model.eval()
    losses: list[float] = []
    own_progress = progress is None
    if progress is None:
        progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), console=console)
        progress.start()
    task = progress.add_task(description, total=batches)
    try:
        for _ in range(batches):
            x, y = get_batch(sequences, targets, windows, batch_size, context, device, generator)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
            losses.append(loss.item())
            progress.advance(task)
    finally:
        progress.remove_task(task)
        if own_progress:
            progress.stop()
    return sum(losses) / len(losses)


def initialize_from_checkpoint(
    model: Feneco, vocabulary: list[str], checkpoint_path: Path | None,
) -> tuple[str | None, int]:
    if checkpoint_path is None:
        return None, 0
    base = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    old_vocabulary = base.get("vocabulary", [])
    old_ids = {character: index for index, character in enumerate(old_vocabulary)}
    new_ids = {character: index for index, character in enumerate(vocabulary)}
    target_state = model.state_dict()
    copied = 0
    for name, source_tensor in base["model_state"].items():
        if name in {"token_embedding.weight", "language_model_head.weight"}:
            target_tensor = target_state[name]
            for character, new_index in new_ids.items():
                old_index = old_ids.get(character)
                if old_index is not None and old_index < source_tensor.shape[0]:
                    target_tensor[new_index].copy_(source_tensor[old_index])
                    if name == "token_embedding.weight":
                        copied += 1
        elif name in target_state and target_state[name].shape == source_tensor.shape:
            target_state[name].copy_(source_tensor)
    model.load_state_dict(target_state)
    return str(checkpoint_path), copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/conversations/pilot/conversations"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/feneco-char-livros-0.3.pt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/feneco-chat-oasst1-0.1.pt"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-batches", type=int, default=30)
    parser.add_argument("--metrics", type=Path, default=Path(".local/chat_experiments.jsonl"))
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.evaluation_batches < 1:
        raise SystemExit("steps, batch-size e evaluation-batches precisam ser positivos.")

    split_records = {
        split: load_records(args.data_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    if any(not split_records[split] for split in split_records):
        raise SystemExit(f"Partição vazia em {args.data_dir}. Execute prepare-chat-data.ps1 primeiro.")
    serialized_train = [serialize_record(row) for row in split_records["train"]]
    vocabulary = sorted({character for text in serialized_train for character in text} | {END_ASSISTANT, UNKNOWN})

    prepared: dict[str, tuple[list[list[int]], list[list[int]], int, list[tuple[int, int]]]] = {}
    for split, records in split_records.items():
        path = args.data_dir / f"{split}.jsonl"
        # Keep parsing identical to the JSONL reader while building character targets.
        texts = [serialize_record(row) for row in records]
        token_to_id = {character: index for index, character in enumerate(vocabulary)}
        sequences: list[list[int]] = []
        labels_by_sequence: list[list[int]] = []
        supervised_characters = 0
        for text, record in zip(texts, records):
            if not text:
                continue
            chars = [token_to_id.get(character, token_to_id[UNKNOWN]) for character in text]
            flags: list[bool] = []
            for message in record.get("messages", []):
                prefix = ROLE_PREFIX.get(str(message.get("role", "")).lower())
                content = str(message.get("content", "")).strip()
                if not prefix or not content:
                    continue
                flags.extend([False] * len(prefix))
                flags.extend([str(message.get("role", "")).lower() == "assistant"] * len(content))
                if str(message.get("role", "")).lower() == "assistant":
                    flags.append(True)  # assistant end marker
                    supervised_characters += len(content) + 1
                flags.append(False)  # newline between turns
            if len(flags) != len(chars):
                raise RuntimeError(f"Template/targets mismatch in {path}.")
            sequences.append(chars[:-1])
            labels_by_sequence.append([
                chars[index + 1] if flags[index + 1] else -100
                for index in range(len(chars) - 1)
            ])
        config_source: dict[str, Any] = {}
        if args.base_checkpoint.exists():
            config_source = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)["model_config"]
        context = int(config_source.get("context_length", 64))
        windows = make_windows(sequences, labels_by_sequence, context)
        prepared[split] = (sequences, labels_by_sequence, supervised_characters, windows)

    base_config: dict[str, Any] = {}
    if args.base_checkpoint.exists():
        base_config = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)["model_config"]
    config_fields = {key: value for key, value in base_config.items() if key in ModelConfig.__dataclass_fields__}
    config_fields.update(vocab_size=len(vocabulary))
    config = ModelConfig(**config_fields)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Feneco(config).to(device)
    base_path, copied_tokens = initialize_from_checkpoint(model, vocabulary, args.base_checkpoint if args.base_checkpoint.exists() else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    counts = count_parameters(model)
    train_seq, train_labels, train_assistant_chars, train_windows = prepared["train"]
    val_seq, val_labels, _, val_windows = prepared["validation"]
    test_seq, test_labels, _, test_windows = prepared["test"]
    train_generator = torch.Generator().manual_seed(args.seed + 1)
    val_generator = torch.Generator().manual_seed(args.seed + 2)
    test_generator = torch.Generator().manual_seed(args.seed + 3)

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="cyan", justify="right")
    summary.add_column()
    summary.add_row("Dispositivo", str(device))
    summary.add_row("Conversas", f"{len(split_records['train'])} treino • {len(split_records['validation'])} validação • {len(split_records['test'])} teste")
    summary.add_row("Respostas do assistente", f"{train_assistant_chars:,} caracteres supervisionados no treino")
    summary.add_row("Contexto", f"{config.context_length} caracteres")
    summary.add_row("Parâmetros", f"{counts['total_parameters']:,} totais • {counts['trainable_parameters']:,} treináveis")
    summary.add_row("Base", Path(base_path).name if base_path else "treino do zero")
    summary.add_row("Caracteres reaproveitados", f"{copied_tokens:,} linhas de embeddings e saída")
    console.print(Panel(summary, title="Feneco • ajuste supervisionado de conversa", border_style="cyan"))

    started = time.perf_counter()
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    recent_losses: list[float] = []
    progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), console=console)
    with progress:
        task = progress.add_task("Ajustando respostas", total=args.steps)
        for step in range(args.steps):
            model.train()
            x, y = get_batch(train_seq, train_labels, train_windows, args.batch_size, config.context_length, device, train_generator)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            recent_losses.append(loss.item())
            progress.advance(task)
            if step == 0 or (step + 1) % 50 == 0 or step + 1 == args.steps:
                validation_loss = estimate_loss(model, val_seq, val_labels, val_windows, args.batch_size, config.context_length, device, val_generator, args.evaluation_batches, progress, "Validando respostas")
                train_loss = sum(recent_losses) / len(recent_losses)
                recent_losses.clear()
                progress.update(task, description=f"Ajuste • passo {step + 1}/{args.steps} • perda treino {train_loss:.4f} • validação {validation_loss:.4f}")
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Nenhum checkpoint foi selecionado pela validação.")
    model.load_state_dict(best_state)
    test_loss = estimate_loss(model, test_seq, test_labels, test_windows, args.batch_size, config.context_length, device, test_generator, args.evaluation_batches)
    duration = time.perf_counter() - started
    model_name = args.checkpoint.stem
    if model_name.lower().startswith("feneco-chat-oasst1"):
        model_name = "Feneco-Chat-OASST1" + model_name[len("feneco-chat-oasst1"):]
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_config": config_to_dict(config),
        "model_state": model.cpu().state_dict(),
        "vocabulary": vocabulary,
        "training_steps": args.steps,
        "training_mode": "assistant_only_character_sft",
        "end_assistant_token": END_ASSISTANT,
        "base_checkpoint": base_path,
        "best_validation_loss": best_validation_loss,
        "test_loss": test_loss,
    }, args.checkpoint)
    record = {
        "timestamp_utc": timestamp_utc(),
        "model_name": model_name,
        "base_checkpoint": base_path,
        "data_path": str(args.data_dir.resolve()),
        "split_conversations": {name: len(rows) for name, rows in split_records.items()},
        "assistant_training_characters": train_assistant_chars,
        "vocabulary_size": len(vocabulary),
        "model_config": config_to_dict(config),
        **counts,
        "training_steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "loss": {"best_validation_assistant_character_loss": best_validation_loss, "held_out_test_assistant_character_loss": test_loss},
        "duration_seconds": duration,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "environment": environment_summary(device),
    }
    append_jsonl(args.metrics, record)
    result = Table.grid(padding=(0, 2))
    result.add_column(style="cyan", justify="right")
    result.add_column()
    result.add_row("Validação", f"perda por caractere assistente {best_validation_loss:.4f}")
    result.add_row("Teste reservado", f"perda por caractere assistente {test_loss:.4f}")
    result.add_row("Checkpoint", str(args.checkpoint))
    result.add_row("Duração", f"{duration:.1f} s")
    console.print(Panel(result, title="Ajuste concluído", border_style="green"))
    console.print(f"Métricas salvas em {args.metrics}")


if __name__ == "__main__":
    main()
