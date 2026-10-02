"""Download small local pilots of Portuguese conversation and text datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Use the Windows certificate store so HTTPS validation respects system-managed roots.
import truststore

truststore.inject_into_ssl()

from datasets import load_dataset
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

console = Console()

OASST = "OpenAssistant/oasst1"
PORTUGUESECHAT = "rishiraj/portuguesechat"
PT_CORPUS = "nicholasKluge/Pt-Corpus-Instruct"
OASST_MIRROR_FILES = {
    "oasst1_train.parquet": "https://hub.oxen.ai/api/repos/OpenAssistant/oasst1/file/main/oasst1_train.parquet",
    "oasst1_validation.parquet": "https://hub.oxen.ai/api/repos/OpenAssistant/oasst1/file/main/oasst1_validation.parquet",
}


def stable_split(key: str, ratios: tuple[int, int, int] = (80, 10, 10)) -> str:
    """Assign a stable split from an example ID, without random state drift."""
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % sum(ratios)
    if bucket < ratios[0]:
        return "train"
    if bucket < ratios[0] + ratios[1]:
        return "validation"
    return "test"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_message(role: str, content: str) -> dict[str, str] | None:
    role_map = {"prompter": "user", "user": "user", "assistant": "assistant", "system": "system"}
    mapped = role_map.get(str(role).lower())
    content = str(content).strip()
    if mapped is None or not content:
        return None
    return {"role": mapped, "content": content}


def download_oasst_mirror(raw_directory: Path) -> list[Path]:
    """Download the public Oxen mirror without contacting Hugging Face."""
    raw_directory.mkdir(parents=True, exist_ok=True)
    paths = []
    with Progress(SpinnerColumn(), TextColumn("Baixando espelho OASST1 • Oxen"), BarColumn(), TaskProgressColumn(), console=console) as progress:
        for filename, url in OASST_MIRROR_FILES.items():
            destination = raw_directory / filename
            if destination.exists() and destination.stat().st_size > 8:
                with destination.open("rb") as existing:
                    valid = existing.read(4) == b"PAR1"
                    existing.seek(-4, 2)
                    valid = valid and existing.read(4) == b"PAR1"
                if valid:
                    paths.append(destination)
                    continue
            temporary = destination.with_suffix(destination.suffix + ".part")
            response = requests.get(url, stream=True, timeout=(20, 120))
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0)) or None
            task = progress.add_task(filename, total=total)
            try:
                with temporary.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
                            progress.advance(task, len(chunk))
                with temporary.open("rb") as downloaded:
                    valid = downloaded.read(4) == b"PAR1"
                    downloaded.seek(-4, 2)
                    valid = valid and downloaded.read(4) == b"PAR1"
                if not valid:
                    raise RuntimeError(f"O arquivo recebido não parece Parquet válido: {filename}")
                temporary.replace(destination)
            finally:
                response.close()
                progress.remove_task(task)
                if temporary.exists():
                    temporary.unlink()
            paths.append(destination)
    return paths


def collect_oasst(parquet_files: list[Path]) -> list[dict[str, Any]]:
    """Read the Oxen Parquet mirror and turn Portuguese branches into chats."""
    stream = load_dataset(
        "parquet", data_files=[str(path) for path in parquet_files],
        split="train", streaming=True,
    )
    nodes: dict[str, dict[str, Any]] = {}
    with Progress(SpinnerColumn(), TextColumn("OASST1 • filtrando português"), BarColumn(), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("Mensagens", total=84_400)
        for row in stream:
            progress.advance(task)
            language = str(row.get("lang", "")).lower().replace("_", "-")
            if language not in {"pt-br", "pt"} or row.get("deleted") or row.get("synthetic"):
                continue
            message = normalize_message(row.get("role", ""), row.get("text", ""))
            if not message:
                continue
            nodes[str(row["message_id"])] = {
                "id": str(row["message_id"]),
                "parent": str(row["parent_id"]) if row.get("parent_id") else None,
                "tree": str(row.get("message_tree_id", "")),
                "message": message,
            }

    children: dict[str, list[str]] = defaultdict(list)
    for node_id, node in nodes.items():
        if node["parent"] in nodes:
            children[node["parent"]].append(node_id)

    examples: list[dict[str, Any]] = []
    for root_id, root in nodes.items():
        if root["parent"] is not None or root["message"]["role"] != "user":
            continue
        stack: list[tuple[str, list[dict[str, str]], tuple[str, ...]]] = [
            (root_id, [root["message"]], (root_id,))
        ]
        while stack:
            node_id, messages, path = stack.pop()
            next_ids = [child for child in children.get(node_id, []) if child not in path]
            if not next_ids:
                if len(messages) >= 2 and messages[-1]["role"] == "assistant":
                    examples.append({
                        "id": f"oasst:{root['tree']}:{node_id}",
                        "source": OASST,
                "source_version": "OASST1 / Oxen public mirror, main branch at download time",
                "license": "Apache-2.0 (dataset card)",
                "mirror": "https://www.oxen.ai/OpenAssistant/oasst1/dir/main/",
                        "split": stable_split(f"oasst:{root['tree']}"),
                        "messages": messages,
                    })
                continue
            for child_id in next_ids:
                child = nodes[child_id]
                if len(messages) < 24:
                    stack.append((child_id, messages + [child["message"]], path + (child_id,)))
    return examples


def dataset_configs(dataset_id: str) -> list[str | None]:
    """Find a usable config from the dataset builder metadata."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(dataset_id, files_metadata=False)
        configs = (info.card_data or {}).get("configs", [])
        names = [item.get("config_name") for item in configs if item.get("config_name")]
        return names or [None]
    except Exception:
        return [None]


def collect_portuguesechat() -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    configs = dataset_configs(PORTUGUESECHAT)
    last_error: Exception | None = None
    for config in configs:
        try:
            for split in ("train_sft", "test_sft"):
                dataset = load_dataset(PORTUGUESECHAT, config, split=split, streaming=True)
                with Progress(SpinnerColumn(), TextColumn(f"PortugueseChat • {split}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
                    task = progress.add_task("Exemplos", total=9_500 if split == "train_sft" else 500)
                    for row in dataset:
                        progress.advance(task)
                        messages = [
                            item for raw in row.get("messages", [])
                            if (item := normalize_message(raw.get("role", ""), raw.get("content", "")))
                        ]
                        if not any(item["role"] == "user" for item in messages) or not any(item["role"] == "assistant" for item in messages):
                            continue
                        prompt_id = str(row.get("prompt_id") or row.get("id") or len(examples))
                        official_test = split == "test_sft"
                        examples.append({
                            "id": f"portuguesechat:{prompt_id}",
                            "source": PORTUGUESECHAT,
                            "source_version": f"{config or 'default'} / {split} snapshot at download time",
                            "license": "CC-BY-NC-4.0 (dataset card)",
                            "split": "test" if official_test else (
                                "train" if stable_split(f"pc:{prompt_id}") != "test" else "validation"
                            ),
                            "messages": messages,
                        })
            return examples
        except Exception as error:
            last_error = error
            examples.clear()
    raise RuntimeError(f"Não consegui ler PortugueseChat ({last_error}).")


def collect_pt_corpus(limit: int) -> list[dict[str, Any]]:
    """Stream only a bounded pilot; never fetch the full 17 GB corpus implicitly."""
    dataset = load_dataset(PT_CORPUS, split="train", streaming=True)
    examples: list[dict[str, Any]] = []
    with Progress(SpinnerColumn(), TextColumn("Pt-Corpus-Instruct • amostra"), BarColumn(), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("Linhas lidas", total=limit)
        for index, row in enumerate(dataset):
            if index >= limit:
                break
            progress.advance(task)
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            origin = str(row.get("metadata", "unknown"))
            examples.append({
                "id": f"pt-corpus:{index}",
                "source": PT_CORPUS,
                "source_version": "train stream prefix sample at download time",
                "license": "Mixed source licenses; see dataset card",
                "split": stable_split(f"pt-corpus:{index}"),
                "kind": "text_sample",
                "source_metadata": origin,
                "text": text,
            })
    return examples


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for example in examples:
            output.write(json.dumps(example, ensure_ascii=False) + "\n")
    return len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/conversations/pilot"))
    parser.add_argument("--sources", nargs="+", choices=("oasst", "portuguesechat", "pt-corpus"), default=["oasst"],
                        help="Sources to prepare. The default uses only the OASST1 Oxen mirror.")
    parser.add_argument("--pt-corpus-rows", type=int, default=2_000,
                        help="Maximum prefix rows to stream from the 17 GB source (default: 2000).")
    args = parser.parse_args()
    if args.pt_corpus_rows < 1:
        raise SystemExit("--pt-corpus-rows precisa ser positivo.")

    console.print(Panel("Preparando conversas em português para o laboratório local.\nO padrão usa o espelho OASST1 no Oxen; os arquivos ficam fora do Git.", title="Feneco • dados de conversa", border_style="cyan"))
    conversations: list[dict[str, Any]] = []
    text_examples: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    source_manifest: list[dict[str, Any]] = []
    try:
        if "oasst" in args.sources:
            raw_paths = download_oasst_mirror(args.output / "raw" / "oasst1")
            rows = collect_oasst(raw_paths)
            conversations.extend(rows)
            source_counts[OASST] = len(rows)
            source_manifest.append({
                "id": OASST,
                "url": f"https://huggingface.co/datasets/{OASST}",
                "mirror": "https://www.oxen.ai/OpenAssistant/oasst1/dir/main/",
                "license": "Apache-2.0 (upstream dataset card)",
                "selection": "Portuguese lang label 'pt' (source does not distinguish Brazilian from European Portuguese); excludes deleted and synthetic; reconstructs user-to-assistant leaf paths",
                "rows_written": len(rows),
                "downloaded_files": [
                    {"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
                    for path in raw_paths
                ],
            })
        if "portuguesechat" in args.sources:
            rows = collect_portuguesechat()
            conversations.extend(rows)
            source_counts[PORTUGUESECHAT] = len(rows)
            source_manifest.append({"id": PORTUGUESECHAT, "url": f"https://huggingface.co/datasets/{PORTUGUESECHAT}", "license": "CC-BY-NC-4.0 (dataset card)", "selection": "train_sft and official test_sft messages; stable 80/10 split of train_sft", "rows_written": len(rows)})
        if "pt-corpus" in args.sources:
            text_examples = collect_pt_corpus(args.pt_corpus_rows)
            source_counts[PT_CORPUS] = len(text_examples)
            source_manifest.append({"id": PT_CORPUS, "url": f"https://huggingface.co/datasets/{PT_CORPUS}", "license": "mixed; see source dataset card", "selection": f"first {args.pt_corpus_rows} streamed training rows only; text samples are not necessarily conversations", "rows_written": len(text_examples), "known_cautions": ["17 GB full dataset", "mixed upstream licenses", "dataset card warns possible personal/sensitive and toxic content", "prefix sample is not representative"]})
    except Exception as error:
        error_text = str(error)
        if "403 Forbidden" in error_text:
            console.print(Panel(
                "A fonte remota retornou HTTP 403. A rede atual bloqueou o acesso; os dados transformados não foram preparados. "
                "Confira a rede e execute novamente.",
                title="Download bloqueado pela rede", border_style="red",
            ))
        else:
            console.print(Panel(error_text, title="Não foi possível obter os dados", border_style="red"))
        raise SystemExit(1) from error

    # Split at conversation/root level before writing, avoiding sibling branches across splits.
    out = args.output
    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        rows = [row for row in conversations if row["split"] == split]
        count = write_jsonl(out / "conversations" / f"{split}.jsonl", rows)
        split_counts.setdefault("conversations", {})[split] = count
        text_rows = [row for row in text_examples if row["split"] == split]
        count = write_jsonl(out / "text_samples" / f"{split}.jsonl", text_rows)
        split_counts.setdefault("text_samples", {})[split] = count

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "local pilot; not yet formatted for Feneco-Char training",
        "sources": source_manifest,
        "split_counts": split_counts,
        "notes": [
            "Every JSONL record retains source, source version, source license and an ID.",
            "The PortugueseChat test split is kept separate. OASST and Pt-Corpus splits are deterministic by conversation tree or record ID.",
            "Do not upload downloaded datasets or trained weights publicly without reviewing each upstream license and the intended use.",
            "This output is a data inspection pilot, not a claim that the current character model can perform assistant fine-tuning.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    table = Table(title="Amostra preparada", show_header=True, header_style="bold cyan")
    table.add_column("Fonte")
    table.add_column("Registros", justify="right")
    table.add_column("Tipo")
    for source, count in source_counts.items():
        kind = "amostra de texto" if source == PT_CORPUS else "conversas em português"
        table.add_row(source, f"{count:,}", kind)
    console.print(table)
    console.print(f"Arquivos locais: [bold]{out.resolve()}[/]")
    console.print("Use `manifest.json` para conferir origem, filtros, contagens e licenças.")


if __name__ == "__main__":
    main()
