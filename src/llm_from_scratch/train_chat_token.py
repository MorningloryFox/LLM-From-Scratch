"""Supervised assistant fine-tuning for a byte-level BPE Feneco checkpoint."""

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
from torch.nn import functional as F

from llm_from_scratch.experiment import append_jsonl, count_parameters, environment_summary, set_seed, timestamp_utc
from llm_from_scratch.model import Feneco, ModelConfig, config_to_dict
from llm_from_scratch.tokenizer import load_tokenizer

console = Console()
ROLE_TOKEN = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}
END_TOKEN = "<|end|>"
PAD_TOKEN = "<|pad|>"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def encode_record(record: dict[str, Any], tokenizer) -> tuple[list[int], list[bool]]:
    ids: list[int] = []; supervised: list[bool] = []
    for message in record.get("messages", []):
        role = str(message.get("role", "")).lower()
        marker = ROLE_TOKEN.get(role)
        content = str(message.get("content", "")).strip()
        if marker is None or not content:
            continue
        ids.append(tokenizer.token_to_id(marker)); supervised.append(False)
        content_ids = tokenizer.encode(content).ids
        ids.extend(content_ids); supervised.extend([role == "assistant"] * len(content_ids))
        if role == "assistant":
            ids.append(tokenizer.token_to_id(END_TOKEN)); supervised.append(True)
        newline_ids = tokenizer.encode("\n").ids
        ids.extend(newline_ids); supervised.extend([False] * len(newline_ids))
    return ids, supervised


def make_batches(records: list[dict[str, Any]], tokenizer, context: int, pad_id: int):
    encoded = []
    for row in records:
        ids, mask = encode_record(row, tokenizer)
        if len(ids) > 1 and any(mask[1:]):
            encoded.append((ids, mask))
    windows = [
        (i, start)
        for i, (ids, mask) in enumerate(encoded)
        for start in range(0, len(ids) - 1, context)
        if any(mask[start + 1:min(start + context + 1, len(ids))])
    ]
    if not windows:
        raise SystemExit("Não há tokens de resposta do assistente para treinar.")
    return encoded, windows


def get_batch(encoded, windows, size, context, pad_id, device, rng):
    selections = torch.randint(len(windows), (size,), generator=rng).tolist()
    xs, ys = [], []
    for choice in selections:
        sequence, start = windows[choice]; ids, mask = sequence
        part = ids[start:start + context + 1]
        flags = mask[start:start + context + 1]
        x = part[:-1]; y = [part[i + 1] if flags[i + 1] else -100 for i in range(len(part) - 1)]
        if len(x) < context:
            missing = context - len(x); x.extend([pad_id] * missing); y.extend([-100] * missing)
        xs.append(x); ys.append(y)
    return torch.tensor(xs, dtype=torch.long, device=device), torch.tensor(ys, dtype=torch.long, device=device)


@torch.no_grad()
def measure(model, encoded, windows, size, context, pad, device, rng, batches):
    model.eval(); losses = []
    for _ in range(batches):
        x, y = get_batch(encoded, windows, size, context, pad, device, rng)
        logits, _ = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100).item())
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/conversations/pilot/conversations"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/feneco-token-livros-0.1.pt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/feneco-chat-oasst1-token-0.1.pt"))
    parser.add_argument("--steps", type=int, default=1000); parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-batches", type=int, default=30); parser.add_argument("--metrics", type=Path, default=Path(".local/chat_experiments.jsonl"))
    args = parser.parse_args()
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    if "tokenizer_json" not in base:
        raise SystemExit("O checkpoint base precisa ser Feneco-Token com tokenizer BPE embutido.")
    tokenizer = load_tokenizer(base["tokenizer_json"])
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    records_by_split = {s: read_jsonl(args.data_dir / f"{s}.jsonl") for s in ("train", "validation", "test")}
    if any(not rows for rows in records_by_split.values()):
        raise SystemExit(f"Partição vazia em {args.data_dir}. Prepare os dados primeiro.")
    prepared = {s: make_batches(rows, tokenizer, base["model_config"]["context_length"], pad_id) for s, rows in records_by_split.items()}
    config = ModelConfig(**base["model_config"])
    set_seed(args.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Feneco(config).to(device); model.load_state_dict(base["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate); counts = count_parameters(model)
    context = config.context_length; rng = torch.Generator().manual_seed(args.seed + 1)
    console.print(Panel(f"Conversas: {len(records_by_split['train'])} treino • {len(records_by_split['validation'])} validação • {len(records_by_split['test'])} teste\nTokenizer: BPE em bytes • vocabulário {config.vocab_size:,}\nContexto: {context} tokens\nParâmetros: {counts['total_parameters']:,}\nBase: {args.base_checkpoint}", title="Feneco-Token • ajuste supervisionado", border_style="cyan"))
    started = time.perf_counter(); best = float("inf"); best_state = None; recent = []
    train_data, train_windows = prepared["train"]; val_data, val_windows = prepared["validation"]
    progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), console=console)
    with progress:
        task = progress.add_task("Ajustando respostas", total=args.steps)
        for step in range(args.steps):
            model.train(); x, y = get_batch(train_data, train_windows, args.batch_size, context, pad_id, device, rng)
            logits, _ = model(x); loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); recent.append(loss.item()); progress.advance(task)
            if step == 0 or (step + 1) % 50 == 0 or step + 1 == args.steps:
                val_loss = measure(model, val_data, val_windows, args.batch_size, context, pad_id, device, torch.Generator().manual_seed(args.seed + 2), args.evaluation_batches)
                progress.update(task, description=f"Ajuste • {step+1}/{args.steps} • treino {sum(recent)/len(recent):.4f} • validação {val_loss:.4f}"); recent.clear()
                if val_loss < best: best = val_loss; best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state is None: raise RuntimeError("Validação não selecionou checkpoint.")
    model.load_state_dict(best_state); test_data, test_windows = prepared["test"]
    test_loss = measure(model, test_data, test_windows, args.batch_size, context, pad_id, device, torch.Generator().manual_seed(args.seed + 3), args.evaluation_batches)
    duration = time.perf_counter() - started; args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**base, "model_state": model.cpu().state_dict(), "training_steps": args.steps, "training_mode": "assistant_only_subword_sft", "base_checkpoint": str(args.base_checkpoint), "best_validation_loss": best, "test_loss": test_loss}, args.checkpoint)
    record = {"timestamp_utc": timestamp_utc(), "model_name": args.checkpoint.stem, "base_checkpoint": str(args.base_checkpoint), "tokenizer": "byte_level_bpe", "vocabulary_size": config.vocab_size, "context_length_tokens": context, "split_conversations": {s: len(rows) for s,rows in records_by_split.items()}, "model_config": config_to_dict(config), **counts, "training_steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "seed": args.seed, "loss":{"best_validation_token":best,"held_out_test_token":test_loss}, "duration_seconds":duration, "checkpoint_path":str(args.checkpoint.resolve()), "checkpoint_size_bytes":args.checkpoint.stat().st_size, "environment":environment_summary(device)}
    append_jsonl(args.metrics, record)
    console.print(Panel(f"Validação (tokens assistente): {best:.4f}\nTeste reservado: {test_loss:.4f}\nCheckpoint: {args.checkpoint}\nDuração: {duration:.1f}s", title="Ajuste concluído", border_style="green"))


if __name__ == "__main__": main()
