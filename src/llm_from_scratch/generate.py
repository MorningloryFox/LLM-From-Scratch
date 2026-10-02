"""Generate text from a locally trained checkpoint."""

import argparse
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from llm_from_scratch.model import Feneco, ModelConfig
from llm_from_scratch.tokenizer import load_tokenizer

console = Console()


def generate_token_checkpoint(args: argparse.Namespace, checkpoint: dict) -> None:
    tokenizer = load_tokenizer(checkpoint["tokenizer_json"])
    model = Feneco(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    end_id = tokenizer.token_to_id("<|end|>")
    role_user = "<|user|>"
    role_assistant = "<|assistant|>"

    def complete(prompt_text: str) -> tuple[str, str]:
        prompt_ids = tokenizer.encode(prompt_text).ids or [0]
        all_ids = model.generate(torch.tensor([prompt_ids], dtype=torch.long), args.tokens, args.temperature)[0].tolist()
        suffix = all_ids[len(prompt_ids):]
        if end_id in suffix:
            suffix = suffix[:suffix.index(end_id)]
        return tokenizer.decode(all_ids, skip_special_tokens=True), tokenizer.decode(suffix, skip_special_tokens=True)

    if args.chat:
        console.print(Panel("Digite uma mensagem e pressione Enter. Use 'sair' para encerrar. O limite de contexto e a geração são contados em tokens BPE.", title="Feneco-Token • sessão local", border_style="bright_cyan"))
        history = ""
        while True:
            try:
                user_text = Prompt.ask("[bold cyan]Você[/]")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Sessão encerrada.[/]"); break
            if user_text.strip().lower() in {"sair", "exit", "quit"}:
                console.print("[dim]Sessão encerrada.[/]"); break
            if not user_text: continue
            if args.assistant_chat:
                prompt = f"{history}{role_user}{user_text}\n{role_assistant}"
            else:
                prompt = f"{history}\n{user_text}" if history else user_text
            _, reply = complete(prompt)
            console.print(Panel(Text(reply), title="Feneco-Token • resposta", border_style="green", padding=(1, 2)))
            history = (prompt + reply + ("<|end|>\n" if args.assistant_chat else "\n"))
            history_ids = tokenizer.encode(history).ids
            if len(history_ids) > model.config.context_length:
                history = tokenizer.decode(history_ids[-model.config.context_length:])
        return

    prompt = f"{role_user}{args.prompt}\n{role_assistant}" if args.assistant_chat else args.prompt
    output, _ = complete(prompt)
    console.print(Panel(Text(output), title="Feneco-Token • texto gerado", border_style="bright_cyan", padding=(1, 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/feneco-char-0.1.pt"))
    parser.add_argument("--prompt", default="")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--chat", action="store_true", help="Open an interactive text-generation session")
    parser.add_argument("--assistant-chat", action="store_true", help="Use the supervised user/assistant turn format")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "tokenizer_json" in checkpoint:
        generate_token_checkpoint(args, checkpoint)
        return
    vocabulary: list[str] = checkpoint["vocabulary"]
    token_to_id = {character: index for index, character in enumerate(vocabulary)}
    model = Feneco(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    end_token = checkpoint.get("end_assistant_token")

    if args.chat:
        console.print(
            Panel(
                "Digite um trecho e pressione Enter. Use 'sair' para encerrar. "
                "O modelo continua o texto com base nos últimos caracteres da conversa.",
                title="Feneco-Char • sessão local",
                border_style="bright_cyan",
            )
        )
        conversation = ""
        while True:
            try:
                user_text = Prompt.ask("[bold cyan]Você[/]")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Sessão encerrada.[/]")
                break
            if user_text.strip().lower() in {"sair", "exit", "quit"}:
                console.print("[dim]Sessão encerrada.[/]")
                break
            if not user_text:
                continue

            if args.assistant_chat:
                prompt_text = f"{conversation}### Usuário:\n{user_text}\n### Assistente:\n"
            else:
                prompt_text = f"{conversation}\n{user_text}" if conversation else user_text
            unknown = sorted(set(prompt_text) - token_to_id.keys())
            if unknown:
                console.print(
                    f"[yellow]Este checkpoint não conhece estes caracteres: {unknown!r}. "
                    "Tente escrever sem eles.[/]"
                )
                continue

            prompt_ids = [token_to_id[character] for character in prompt_text]
            generated = model.generate(
                torch.tensor([prompt_ids], dtype=torch.long), args.tokens, args.temperature
            )[0].tolist()
            generated_text = "".join(vocabulary[index] for index in generated[len(prompt_ids):])
            if end_token and end_token in generated_text:
                reply = generated_text.split(end_token, 1)[0]
            else:
                reply = generated_text
            console.print(
                Panel(Text(reply), title="Feneco-Char • continuação", border_style="green", padding=(1, 2))
            )
            ending = end_token if args.assistant_chat and end_token else ""
            conversation = (prompt_text + reply + ending + "\n")[-model.config.context_length:]
        return

    prompt = f"### Usuário:\n{args.prompt}\n### Assistente:\n" if args.assistant_chat else args.prompt
    unknown = sorted(set(prompt) - token_to_id.keys())
    if unknown:
        raise SystemExit(f"Prompt contains characters absent from training corpus: {unknown!r}")
    prompt_ids = [token_to_id[character] for character in prompt]
    if not prompt_ids:
        prompt_ids = [0]
    generated = model.generate(
        torch.tensor([prompt_ids], dtype=torch.long), args.tokens, args.temperature
    )
    output = "".join(vocabulary[index] for index in generated[0].tolist())
    if end_token and end_token in output[len(prompt):]:
        output = output[: len(prompt)] + output[len(prompt):].split(end_token, 1)[0]
    console.print(Panel(Text(output), title="Feneco-Char • texto gerado", border_style="bright_cyan", padding=(1, 2)))


if __name__ == "__main__":
    main()
