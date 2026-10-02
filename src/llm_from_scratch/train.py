"""Train Feneco-Char locally and save a reproducible experiment record."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

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
from llm_from_scratch.model import Feneco, ModelConfig, config_to_dict


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
) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, batch_size, context_length, device, generator)
        _, loss = model(x, y)
        assert loss is not None
        losses.append(loss.item())
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="UTF-8 text corpus")
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
    if not args.data.is_file():
        raise SystemExit(f"Corpus not found: {args.data}")

    text = args.data.read_text(encoding="utf-8")
    if not text:
        raise SystemExit("The corpus is empty.")
    vocabulary = sorted(set(text))
    token_to_id = {character: index for index, character in enumerate(vocabulary)}
    encoded = torch.tensor([token_to_id[character] for character in text], dtype=torch.long)
    train_data, validation_data, test_data = split_corpus(encoded, args.context_length)

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

    print(f"device={device} corpus_characters={len(encoded)} vocabulary={len(vocabulary)}")
    print(
        f"layers={config.number_of_layers} heads_per_layer={config.number_of_heads} "
        f"attention_head_instances={head_instances} "
        f"total_parameters={counts['total_parameters']} "
        f"trainable_parameters={counts['trainable_parameters']}"
    )
    print(
        f"allowed_causal_attention_pairs_at_context={allowed_attention_pairs} "
        "(position pairs per full context across all heads and layers)"
    )

    started = time.perf_counter()
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    recent_losses: list[float] = []
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

        if step == 0 or (step + 1) % 50 == 0 or step + 1 == args.steps:
            validation_loss = estimate_loss(
                model,
                validation_data,
                args.batch_size,
                args.context_length,
                device,
                args.seed + 2,
                args.evaluation_batches,
            )
            train_loss = sum(recent_losses) / len(recent_losses)
            recent_losses.clear()
            print(
                f"step={step + 1}/{args.steps} train_loss={train_loss:.4f} "
                f"validation_loss={validation_loss:.4f}"
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
        },
        args.checkpoint,
    )
    print(f"best_validation_loss={best_validation_loss:.4f} test_loss={test_loss:.4f}")
    print(f"checkpoint saved to {args.checkpoint}")

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
    print(f"experiment metrics appended to {args.metrics}")


if __name__ == "__main__":
    main()
