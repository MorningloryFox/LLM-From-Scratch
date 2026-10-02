"""Generate text from a locally trained checkpoint."""

import argparse
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from llm_from_scratch.model import Feneco, ModelConfig

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/feneco-char-0.1.pt"))
    parser.add_argument("--prompt", default="")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vocabulary: list[str] = checkpoint["vocabulary"]
    token_to_id = {character: index for index, character in enumerate(vocabulary)}
    unknown = sorted(set(args.prompt) - token_to_id.keys())
    if unknown:
        raise SystemExit(f"Prompt contains characters absent from training corpus: {unknown!r}")

    model = Feneco(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    prompt_ids = [token_to_id[character] for character in args.prompt]
    if not prompt_ids:
        prompt_ids = [0]
    generated = model.generate(
        torch.tensor([prompt_ids], dtype=torch.long), args.tokens, args.temperature
    )
    output = "".join(vocabulary[index] for index in generated[0].tolist())
    console.print(Panel(Text(output), title="Feneco-Char • texto gerado", border_style="bright_cyan", padding=(1, 2)))


if __name__ == "__main__":
    main()
