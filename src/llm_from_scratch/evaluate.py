"""Evaluate a Feneco checkpoint on its held-out test split."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_from_scratch.experiment import count_parameters, corpus_sha256, set_seed
from llm_from_scratch.corpus import load_documents, split_documents
from llm_from_scratch.model import Feneco, ModelConfig
from llm_from_scratch.train import estimate_loss, split_corpus

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, required=True,
        help="Same UTF-8 .txt file or folder used to train",
    )
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()
    if args.batches < 1 or args.batch_size < 1:
        raise SystemExit("batches and batch-size must be positive")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    expected_digest = checkpoint.get("corpus_sha256")
    if expected_digest and corpus_sha256(args.data) != expected_digest:
        raise SystemExit(
            "Corpus SHA-256 differs from the file used to train this checkpoint. "
            "Use the original corpus to evaluate its held-out split."
        )
    vocabulary: list[str] = checkpoint["vocabulary"]
    documents = load_documents(args.data)
    text = "\n\n".join(content for _, content in documents)
    unknown = sorted(set(text) - set(vocabulary))
    if unknown:
        raise SystemExit(
            f"Corpus has characters absent from checkpoint vocabulary: {unknown[:20]!r}"
        )
    token_to_id = {character: index for index, character in enumerate(vocabulary)}
    config = ModelConfig(**checkpoint["model_config"])
    if args.data.is_dir():
        split_texts, split_files = split_documents(
            documents, config.context_length, int(checkpoint["seed"])
        )
        test_text = split_texts[2]
        test_book_count = len(split_files[2])
    else:
        encoded = torch.tensor(
            [token_to_id[character] for character in text], dtype=torch.long
        )
        _, _, test_data = split_corpus(encoded, config.context_length)
        test_text = None
        test_book_count = 0
    if test_text is not None:
        test_data = torch.tensor(
            [token_to_id[character] for character in test_text], dtype=torch.long
        )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Feneco(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loss = estimate_loss(
        model,
        test_data,
        args.batch_size,
        config.context_length,
        device,
        args.seed,
        args.batches,
    )
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", justify="right")
    table.add_column()
    table.add_row("Dispositivo", str(device))
    table.add_row("Perda de teste", f"{loss:.4f}")
    if args.data.is_dir():
        table.add_row("Livros reservados para teste", str(test_book_count))
    table.add_row("Perplexidade", f"{math.exp(loss):.4f}")
    table.add_row("Parâmetros", f"{count_parameters(model)['total_parameters']:,}")
    table.add_row("Arquitetura", f"{config.number_of_layers} camadas • {config.number_of_heads} cabeças por camada • sem quantização")
    console.print(Panel(table, title="[bold bright_cyan]Avaliação do Feneco[/]", border_style="cyan"))
    console.print(Panel("Métrica de linguagem por caractere; não mede qualidade de pesquisa web nem de respostas.", border_style="yellow", title="Limite da métrica"))


if __name__ == "__main__":
    main()
