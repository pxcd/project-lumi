
"""Dataset preparation and batching for Project Lumi."""

import json
import random
from pathlib import Path

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


class LumiDataset:
    """Prepare text corpora for Lumi's next-token training."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        vocab_size: int = 8192,
        context_length: int = 512,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.seed = seed

        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = None
        self.train_tokens = None
        self.val_tokens = None
        self.train_rng = np.random.default_rng(seed)
        self.val_rng = np.random.default_rng(seed + 1)

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize line endings and remove null bytes."""
        return (
            text.replace("\r\n", "\n")
            .replace("\r", "\n")
            .replace("\x00", "")
        )

    def load_documents(
        self,
        corpus_dir: str | Path,
    ) -> list[str]:
        """Load UTF-8 text files in deterministic order."""
        corpus_dir = Path(corpus_dir)

        paths = sorted(corpus_dir.rglob("*.txt"))

        if not paths:
            raise ValueError(
                f"No .txt files found in {corpus_dir}"
            )

        documents = []

        for path in paths:
            text = self.clean_text(
                path.read_text(encoding="utf-8")
            )

            if text.strip():
                documents.append(text)

        if len(documents) < 2:
            raise ValueError(
                "At least two nonempty documents are required"
            )

        return documents

    def split_documents(
        self,
        documents: list[str],
        validation_fraction: float = 0.1,
    ) -> tuple[list[str], list[str]]:
        """Split documents before tokenizer training."""
        if not 0 < validation_fraction < 1:
            raise ValueError(
                "validation_fraction must be between 0 and 1"
            )

        if len(documents) < 2:
            raise ValueError(
                "At least two documents are required"
            )

        shuffled = documents.copy()
        random.Random(self.seed).shuffle(shuffled)

        val_count = max(
            1,
            round(len(shuffled) * validation_fraction),
        )

        val_count = min(val_count, len(shuffled) - 1)

        return shuffled[val_count:], shuffled[:val_count]

    def train_tokenizer(
        self,
        train_documents: list[str],
    ) -> Tokenizer:
        """Train a byte-level BPE tokenizer from scratch."""
        tokenizer = Tokenizer(models.BPE(unk_token=None))

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False
        )

        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[],
            show_progress=True,
        )

        tokenizer.train_from_iterator(
            train_documents,
            trainer=trainer,
            length=len(train_documents),
        )

        actual_size = tokenizer.get_vocab_size()

        if actual_size != self.vocab_size:
            raise ValueError(
                f"Tokenizer produced {actual_size} tokens; "
                f"expected {self.vocab_size}. "
                "Use a larger training corpus."
            )

        self.tokenizer = tokenizer

        tokenizer.save(
            str(self.data_dir / "tokenizer.json")
        )

        return tokenizer

    def encode_documents(
        self,
        documents: list[str],
    ) -> np.ndarray:
        """Encode documents into a contiguous token stream."""
        if self.tokenizer is None:
            raise RuntimeError(
                "Train or load the tokenizer first"
            )

        token_ids = []

        for document in documents:
            token_ids.extend(
                self.tokenizer.encode(document).ids
            )

        if not token_ids:
            raise ValueError("No tokens produced")

        return np.asarray(token_ids, dtype=np.uint16)

    def prepare(
        self,
        corpus_dir: str | Path,
        validation_fraction: float = 0.1,
    ) -> dict:
        """Prepare and save training/validation token streams."""
        documents = self.load_documents(corpus_dir)

        train_docs, val_docs = self.split_documents(
            documents,
            validation_fraction,
        )

        self.train_tokenizer(train_docs)

        self.train_tokens = self.encode_documents(train_docs)
        self.val_tokens = self.encode_documents(val_docs)

        self.train_tokens.tofile(
            self.data_dir / "train.bin"
        )

        self.val_tokens.tofile(
            self.data_dir / "val.bin"
        )

        metadata = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "seed": self.seed,
            "train_documents": len(train_docs),
            "validation_documents": len(val_docs),
            "train_tokens": len(self.train_tokens),
            "validation_tokens": len(self.val_tokens),
        }

        (self.data_dir / "dataset.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        return metadata

    def load_prepared(self) -> None:
        """Load a previously prepared dataset."""
        self.tokenizer = Tokenizer.from_file(
            str(self.data_dir / "tokenizer.json")
        )

        self.train_tokens = np.fromfile(
            self.data_dir / "train.bin",
            dtype=np.uint16,
        )

        self.val_tokens = np.fromfile(
            self.data_dir / "val.bin",
            dtype=np.uint16,
        )

        metadata = json.loads(
            (self.data_dir / "dataset.json").read_text(
                encoding="utf-8"
            )
        )

        if self.tokenizer.get_vocab_size() != metadata["vocab_size"]:
            raise ValueError("Tokenizer vocabulary mismatch")

        if self.context_length != metadata["context_length"]:
            raise ValueError("Context length mismatch")

    def get_batch(
        self,
        split: str = "train",
        batch_size: int = 8,
    ) -> mx.array:
        """Sample random contiguous next-token training windows."""
        if split not in ("train", "val"):
            raise ValueError(
                "split must be 'train' or 'val'"
            )

        tokens = (
            self.train_tokens
            if split == "train"
            else self.val_tokens
        )

        if tokens is None:
            raise RuntimeError(
                "Dataset has not been prepared or loaded"
            )

        # Model receives context_length input tokens
        # plus one target token.
        window_size = self.context_length + 1

        if len(tokens) < window_size:
            raise ValueError(
                f"{split} split has {len(tokens)} tokens; "
                f"need at least {window_size}"
            )

        rng = self.train_rng if split == "train" else self.val_rng

        starts = rng.integers(
            0,
            len(tokens) - window_size + 1,
            size=batch_size,
        )

        batch = np.stack([
            tokens[start:start + window_size]
            for start in starts
        ])

        return mx.array(
            batch.astype(np.int32)
        )
