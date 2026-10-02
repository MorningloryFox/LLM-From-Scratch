"""Train a byte-level BPE Feneco model on a local text corpus."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from llm_from_scratch.corpus import load_documents, split_documents
from llm_from_scratch.experiment import append_jsonl, count_parameters, corpus_sha256, environment_summary, git_revision, git_working_tree_dirty, set_seed, timestamp_utc
from llm_from_scratch.model import Feneco, ModelConfig, config_to_dict
from llm_from_scratch.tokenizer import SPECIAL_TOKENS, train_tokenizer

console = Console()


def batch(data: torch.Tensor, size: int, context: int, device: torch.device, rng: torch.Generator):
    starts = torch.randint(len(data) - context - 1, (size,), generator=rng)
    x = torch.stack([data[i:i + context] for i in starts]).to(device)
    y = torch.stack([data[i + 1:i + context + 1] for i in starts]).to(device)
    return x, y


@torch.no_grad()
def measure(model, data, size, context, device, seed, batches):
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(batches):
        x, y = batch(data, size, context, device, rng)
        _, loss = model(x, y)
        values.append(loss.item())
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-batches", type=int, default=20)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/feneco-token-0.1.pt"))
    parser.add_argument("--metrics", type=Path, default=Path(".local/experiments.jsonl"))
    args = parser.parse_args()
    if min(args.steps, args.context_length, args.vocab_size, args.batch_size, args.evaluation_batches) < 1:
        raise SystemExit("Passos, contexto, vocabulário, lote e avaliações devem ser positivos.")

    documents = load_documents(args.data)
    if args.data.is_dir():
        split_texts, split_files = split_documents(documents, args.context_length, args.seed)
        train_text, val_text, test_text = split_texts
        split_strategy = "whole_documents"
    else:
        text = documents[0][1]
        a, b = int(len(text) * .8), int(len(text) * .9)
        train_text, val_text, test_text = text[:a], text[a:b], text[b:]
        if min(map(len, (train_text, val_text, test_text))) < args.context_length + 2:
            raise SystemExit("Corpus curto para contexto solicitado; use outro corpus ou reduza o contexto.")
        split_files = ((), (), ())
        split_strategy = "contiguous_text_segments"

    tokenizer = train_tokenizer([train_text], args.vocab_size)
    train_ids, val_ids, test_ids = [
        torch.tensor(tokenizer.encode(part).ids, dtype=torch.long)
        for part in (train_text, val_text, test_text)
    ]
    if any(len(part) < args.context_length + 2 for part in (train_ids, val_ids, test_ids)):
        raise SystemExit(
            "Uma divisão ficou curta depois da tokenização. Use mais textos ou reduza -ContextLength."
        )
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig(vocab_size=tokenizer.get_vocab_size(), context_length=args.context_length)
    model = Feneco(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    counts = count_parameters(model)
    rng = torch.Generator().manual_seed(args.seed + 1)
    table = Table.grid(padding=(0, 2)); table.add_column(style="cyan", justify="right"); table.add_column()
    table.add_row("Dispositivo", str(device))
    table.add_row("Corpus", f"{sum(map(len, (train_text, val_text, test_text))):,} caracteres → {len(train_ids)+len(val_ids)+len(test_ids):,} tokens")
    table.add_row("Tokenizer", f"BPE em bytes • vocabulário {config.vocab_size:,}")
    table.add_row("Contexto", f"{config.context_length} tokens")
    table.add_row("Arquitetura", f"{config.number_of_layers} camadas • {config.number_of_heads} cabeças/camada")
    table.add_row("Parâmetros", f"{counts['total_parameters']:,}")
    console.print(Panel(table, title="Feneco-Token • treino", border_style="cyan"))

    started = time.perf_counter(); best = float("inf"); best_state = None; recent = []
    progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), console=console)
    with progress:
        task = progress.add_task("Treinando", total=args.steps)
        for step in range(args.steps):
            model.train(); x, y = batch(train_ids, args.batch_size, args.context_length, device, rng)
            _, loss = model(x, y); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            recent.append(loss.item()); progress.advance(task)
            if step == 0 or (step + 1) % 50 == 0 or step + 1 == args.steps:
                val_loss = measure(model, val_ids, args.batch_size, args.context_length, device, args.seed + 2, args.evaluation_batches)
                progress.update(task, description=f"Treino • {step+1}/{args.steps} • perda {sum(recent)/len(recent):.4f} • validação {val_loss:.4f}"); recent.clear()
                if val_loss < best:
                    best = val_loss; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None: raise RuntimeError("Validação não selecionou checkpoint.")
    model.load_state_dict(best_state)
    test_loss = measure(model, test_ids, args.batch_size, args.context_length, device, args.seed + 3, args.evaluation_batches)
    duration = time.perf_counter() - started; digest = corpus_sha256(args.data)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_config": config_to_dict(config), "model_state": model.cpu().state_dict(), "tokenizer_json": tokenizer.to_str(), "tokenizer_type": "byte_level_bpe", "training_steps": args.steps, "seed": args.seed, "best_validation_loss": best, "test_loss": test_loss, "corpus_sha256": digest, "split_strategy": split_strategy, "split_files": {k: list(v) for k,v in zip(("train","validation","test"), split_files)}}, args.checkpoint)
    record = {"timestamp_utc": timestamp_utc(), "model_name": args.checkpoint.stem, "code_revision": git_revision(), "code_working_tree_dirty": git_working_tree_dirty(), "corpus_path": str(args.data.resolve()), "corpus_sha256": digest, "corpus_characters": sum(map(len,(train_text,val_text,test_text))), "tokenizer": "byte_level_bpe", "vocabulary_size": config.vocab_size, "split_strategy": split_strategy, "split_files": {k:list(v) for k,v in zip(("train","validation","test"),split_files)}, "model_config": config_to_dict(config), **counts, "attention_head_instances": config.number_of_layers*config.number_of_heads, "quantization": "none", "training": {"steps":args.steps,"batch_size":args.batch_size,"learning_rate":args.learning_rate,"seed":args.seed,"context_length_tokens":args.context_length,"evaluation_batches":args.evaluation_batches}, "loss":{"best_validation":best,"held_out_test":test_loss},"duration_seconds":duration,"checkpoint_path":str(args.checkpoint.resolve()),"checkpoint_size_bytes":args.checkpoint.stat().st_size,"environment":environment_summary(device)}
    append_jsonl(args.metrics, record)
    console.print(Panel(f"Validação: {best:.4f}\nTeste reservado: {test_loss:.4f}\nCheckpoint: {args.checkpoint}\nDuração: {duration:.1f}s", title="Treino concluído", border_style="green"))


if __name__ == "__main__":
    main()
