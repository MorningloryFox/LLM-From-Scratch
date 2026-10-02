"""Train the small model on a local text file."""

import argparse
from pathlib import Path

import torch

from llm_from_scratch.model import Mirim, ModelConfig, config_to_dict


def get_batch(
    data: torch.Tensor, batch_size: int, context_length: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(len(data) - context_length - 1, (batch_size,))
    x = torch.stack([data[start : start + context_length] for start in starts])
    y = torch.stack([data[start + 1 : start + context_length + 1] for start in starts])
    return x.to(device), y.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="UTF-8 text corpus")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mirim-0.1.pt"))
    args = parser.parse_args()

    text = args.data.read_text(encoding="utf-8")
    if not text:
        raise SystemExit("The corpus is empty.")
    vocabulary = sorted(set(text))
    minimum_length = 2 * (args.context_length + 2)
    if len(text) < minimum_length:
        raise SystemExit(
            "The corpus must contain at least "
            f"2 * (context_length + 2) characters ({minimum_length} for this run)."
        )

    token_to_id = {character: index for index, character in enumerate(vocabulary)}
    encoded = torch.tensor([token_to_id[character] for character in text], dtype=torch.long)
    split = max(args.context_length + 2, int(0.9 * len(encoded)))
    split = min(split, len(encoded) - args.context_length - 2)
    train_data = encoded[:split]
    validation_data = encoded[split:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig(vocab_size=len(vocabulary), context_length=args.context_length)
    model = Mirim(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    attention_head_instances = config.number_of_layers * config.number_of_heads
    causal_attention_pairs = (
        attention_head_instances * config.context_length * (config.context_length + 1) // 2
    )

    print(f"device={device} characters={len(encoded)} vocabulary={len(vocabulary)}")
    print(
        f"layers={config.number_of_layers} heads_per_layer={config.number_of_heads} "
        f"attention_head_instances={attention_head_instances} "
        f"trainable_parameters={parameter_count}"
    )
    print(
        f"causal_attention_pairs_at_context={causal_attention_pairs} "
        "(allowed query-key position pairs per full context)"
    )
    for step in range(args.steps):
        model.train()
        x, y = get_batch(train_data, 16, args.context_length, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == args.steps:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(validation_data, 16, args.context_length, device)
                _, validation_loss = model(vx, vy)
            print(
                f"step={step + 1}/{args.steps} train_loss={loss.item():.3f} "
                f"validation_loss={validation_loss.item():.3f}"
            )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": config_to_dict(config),
            "model_state": model.cpu().state_dict(),
            "vocabulary": vocabulary,
            "training_steps": args.steps,
            "trainable_parameters": parameter_count,
        },
        args.checkpoint,
    )
    print(f"checkpoint saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
