"""Extract selectable text from one PDF or a folder of PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from pypdf import PdfReader
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

console = Console()


def find_pdfs(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() != ".pdf":
            raise SystemExit(f"O arquivo informado não é PDF: {source}")
        return [source]
    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")
    raise SystemExit(f"Caminho não encontrado: {source}")


def extract_one(
    pdf_path: Path,
    relative_path: Path,
    output_root: Path,
    overwrite: bool,
    delete_pdf: bool,
) -> dict[str, object]:
    text_path = output_root / "txt" / relative_path.with_suffix(".txt")
    metadata_path = output_root / "json" / relative_path.with_name(
        f"{relative_path.stem}.extraction.json"
    )
    if text_path.exists() and not overwrite:
        return {"pdf": str(pdf_path), "status": "Ignorado (TXT já existe)", "characters": 0}

    reader = PdfReader(pdf_path)
    page_texts: list[str] = []
    empty_pages = 0
    for page in reader.pages:
        extracted = page.extract_text() or ""
        extracted = extracted.replace("\x00", "")
        if not extracted.strip():
            empty_pages += 1
        page_texts.append(extracted.rstrip())

    text = "\n\n".join(page_texts).strip() + "\n"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text, encoding="utf-8")
    metadata = {
        "source_pdf": str(pdf_path.resolve()),
        "source_pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "output_txt": str(text_path.resolve()),
        "title": (reader.metadata.title if reader.metadata else None),
        "author": (reader.metadata.author if reader.metadata else None),
        "pages": len(reader.pages),
        "pages_without_extractable_text": empty_pages,
        "text_characters": len(text),
        "extractor": "pypdf",
        "extractor_version": version("pypdf"),
        "extracted_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Extração automática; revisar o TXT antes de usar no corpus. Não executa OCR.",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    deleted = False
    if delete_pdf and text.strip():
        pdf_path.unlink()
        deleted = True
    return {
        "pdf": str(pdf_path),
        "status": (
            "Extraído; PDF excluído" if deleted else
            "Extraído" if text.strip() else "Sem texto extraível; PDF preservado"
        ),
        "characters": len(text),
        "pages": len(reader.pages),
        "empty_pages": empty_pages,
        "txt": str(text_path),
        "metadata": str(metadata_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path", type=Path, default=Path("data"),
        help="PDF individual ou pasta (inclui subpastas; padrão: data)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Substitui TXT e metadados de extração que já existirem",
    )
    parser.add_argument(
        "--delete-pdf", action="store_true",
        help="Apaga o PDF somente após salvar TXT e JSON com texto extraído",
    )
    args = parser.parse_args()
    source = args.path
    pdfs = find_pdfs(source)
    if not pdfs:
        console.print(Panel(f"Nenhum PDF encontrado em {args.path}.", title="Extração de livros", border_style="yellow"))
        return
    input_root = source if source.is_dir() else source.parent
    output_root = input_root

    rows: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
        TaskProgressColumn(), console=console,
    ) as progress:
        task = progress.add_task("Extraindo PDFs", total=len(pdfs))
        for pdf in pdfs:
            progress.update(task, description=f"Extraindo {pdf.name}")
            try:
                relative_path = pdf.relative_to(input_root)
                rows.append(
                    extract_one(
                        pdf, relative_path, output_root, args.overwrite, args.delete_pdf
                    )
                )
            except Exception as error:
                failures.append((str(pdf), str(error)))
            progress.advance(task)

    table = Table(title="Resultado da extração", header_style="bold cyan", show_lines=True)
    table.add_column("PDF")
    table.add_column("Resultado")
    table.add_column("Páginas", justify="right")
    table.add_column("Caracteres", justify="right")
    for row in rows:
        table.add_row(
            str(row["pdf"]), str(row["status"]), str(row.get("pages", "—")),
            f"{int(row['characters']):,}",
        )
    console.print(table)
    if failures:
        errors = Table(title="Arquivos com erro", header_style="bold red")
        errors.add_column("PDF")
        errors.add_column("Erro")
        for pdf, message in failures:
            errors.add_row(pdf, message)
        console.print(errors)
    console.print(
        Panel(
            "Confira os TXT antes de treinar. PDFs digitalizados como imagem podem exigir OCR; "
            "esta rotina só extrai a camada de texto existente.",
            title="Próximo passo", border_style="yellow",
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
