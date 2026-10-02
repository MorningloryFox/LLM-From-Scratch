"""Show a compact summary of local training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path(".local/experiments.jsonl"))
    args = parser.parse_args()
    if not args.metrics.is_file():
        raise SystemExit(f"No local experiment log found at {args.metrics}")

    print(
        "date (UTC)             model             params    heads/layers "
        "steps   val loss  test loss  seconds"
    )
    for line in args.metrics.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        run = json.loads(line)
        config = run["model_config"]
        losses = run["loss"]
        print(
            f"{run['timestamp_utc']:<22} {run['model_name']:<17} "
            f"{run['total_parameters']:>8} "
            f"{config['number_of_heads']}/{config['number_of_layers']:<11} "
            f"{run['training']['steps']:<7} "
            f"{losses['best_validation']:<9.4f} "
            f"{losses['held_out_test']:<10.4f} "
            f"{run['duration_seconds']:.1f}"
        )


if __name__ == "__main__":
    main()
