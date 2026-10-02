"""Load and split local text documents for reproducible experiments."""

from __future__ import annotations

import random
from pathlib import Path


def load_documents(source: Path) -> list[tuple[str, str]]:
    """Load one UTF-8 text file or all UTF-8 text files in a directory."""
    if source.is_file():
        if source.suffix.lower() != ".txt":
            raise SystemExit(f"Corpus file must be a .txt file: {source}")
        paths = [source]
        root = source.parent
    elif source.is_dir():
        paths = sorted(path for path in source.rglob("*.txt") if path.is_file())
        root = source
    else:
        raise SystemExit(f"Corpus not found: {source}")

    if not paths:
        raise SystemExit(f"No .txt files found in corpus path: {source}")

    documents = []
    for path in paths:
        relative_path = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        if text:
            documents.append((relative_path, text))
    if not documents:
        raise SystemExit("The corpus is empty.")
    return documents


def split_documents(
    documents: list[tuple[str, str]], context_length: int, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Split whole documents into reproducible train/validation/test groups."""
    if len(documents) < 3:
        raise SystemExit(
            "Book-level splitting needs at least 3 non-empty .txt files. "
            "Add more books or train from one combined .txt file."
        )

    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    train_count = min(max(1, round(0.8 * len(shuffled))), len(shuffled) - 2)
    validation_count = min(
        max(1, round(0.1 * len(shuffled))), len(shuffled) - train_count - 1
    )
    groups = (
        shuffled[:train_count],
        shuffled[train_count : train_count + validation_count],
        shuffled[train_count + validation_count :],
    )
    names = tuple(tuple(name for name, _ in group) for group in groups)
    texts = tuple("\n\n".join(text for _, text in group) for group in groups)
    labels = ("training", "validation", "test")
    minimum = context_length + 2
    for label, text in zip(labels, texts):
        if len(text) < minimum:
            raise SystemExit(
                f"The {label} document split has only {len(text)} characters; "
                f"it needs at least {minimum}. Add or redistribute books, "
                "or use a shorter context length."
            )
    return texts, names
