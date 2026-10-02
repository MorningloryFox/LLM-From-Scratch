"""Byte-level BPE tokenizer used by Feneco-Token checkpoints."""

from __future__ import annotations

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<|unk|>", "<|pad|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]


def train_tokenizer(texts: list[str], vocab_size: int = 1024) -> Tokenizer:
    """Fit byte-level BPE on training texts; bytes make UTF-8 text lossless."""
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def load_tokenizer(serialized: str) -> Tokenizer:
    return Tokenizer.from_str(serialized)
