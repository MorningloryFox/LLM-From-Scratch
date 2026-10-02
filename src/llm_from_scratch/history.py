"""Show a compact summary of local training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path(".local/experiments.jsonl"))
    args = parser.parse_args()
    if not args.metrics.is_file():
        raise SystemExit(f"No local experiment log found at {args.metrics}")

    table = Table(title="Experimentos locais", header_style="bold cyan", show_lines=True)
    table.add_column("Data (UTC)")
    table.add_column("Modelo")
    table.add_column("Parâmetros", justify="right")
    table.add_column("Cabeças/camadas", justify="center")
    table.add_column("Passos", justify="right")
    table.add_column("Validação", justify="right")
    table.add_column("Teste", justify="right")
    table.add_column("Tempo (s)", justify="right")
    count = 0
    for line in args.metrics.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        run = json.loads(line)
        config = run["model_config"]
        losses = run["loss"]
        table.add_row(
            run["timestamp_utc"], run["model_name"], f"{run['total_parameters']:,}",
            f"{config['number_of_heads']}/{config['number_of_layers']}",
            str(run["training"]["steps"]), f"{losses['best_validation']:.4f}",
            f"{losses['held_out_test']:.4f}", f"{run['duration_seconds']:.1f}",
        )
        count += 1
    if count:
        console.print(table)
    else:
        console.print(Panel("Ainda não há execuções registradas.", title="Histórico local", border_style="yellow"))


if __name__ == "__main__":
    main()
