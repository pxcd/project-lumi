
"""Dataset preparation and batching for Project Lumi."""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

from lumi.artifacts import fingerprint_files, sha256_file, verify_fingerprints


RAW_STORY_DELIMITER = "<|endoftext|>"
END_OF_STORY_TOKEN = "<|endofstory|>"


@dataclass(frozen=True)
class StoryGroup:
    """Stories belonging to one historical v1 corpus group."""

    name: str
    stories: tuple[str, ...]
    source_sha256: str


class LumiDataset:
    """Prepare text corpora for Lumi's next-token training."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        vocab_size: int = 8192,
        context_length: int = 512,
        seed: int = 42,
        eos_token: str = END_OF_STORY_TOKEN,
    ):
        self.data_dir = Path(data_dir)
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.seed = seed
        self.eos_token = eos_token

        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = None
        self.train_tokens = None
        self.val_tokens = None
        self.metadata = None
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

    def load_complete_stories(
        self,
        raw_file: str | Path,
    ) -> tuple[list[str], dict]:
        """Load only delimiter-terminated stories from a raw TinyStories file."""
        raw_file = Path(raw_file)
        text = self.clean_text(raw_file.read_text(encoding="utf-8"))
        parts = text.split(RAW_STORY_DELIMITER)

        if len(parts) == 1:
            raise ValueError(
                f"Raw source contains no {RAW_STORY_DELIMITER} delimiters"
            )

        complete_parts = parts[:-1]
        stories = [part.strip() for part in complete_parts if part.strip()]
        trailing_fragment = parts[-1].strip()

        if len(stories) < 2:
            raise ValueError("At least two complete stories are required")

        return stories, {
            "path": raw_file.name,
            "sha256": sha256_file(raw_file),
            "delimiter": RAW_STORY_DELIMITER,
            "delimiter_count": len(parts) - 1,
            "complete_stories": len(stories),
            "empty_records": sum(
                1 for part in complete_parts if not part.strip()
            ),
            "dropped_trailing_fragment": bool(trailing_fragment),
            "dropped_trailing_characters": len(trailing_fragment),
        }

    def load_story_groups(
        self,
        raw_file: str | Path,
        reference_corpus_dir: str | Path,
    ) -> tuple[list[StoryGroup], dict]:
        """Rebuild exact v1 corpus groups from delimiter-terminated stories."""
        stories, raw_metadata = self.load_complete_stories(raw_file)
        reference_corpus_dir = Path(reference_corpus_dir)
        paths = sorted(reference_corpus_dir.rglob("*.txt"))

        if not paths:
            raise ValueError(
                f"No reference .txt files found in {reference_corpus_dir}"
            )

        groups = []
        story_index = 0

        for path in paths:
            reference = self.clean_text(path.read_text(encoding="utf-8"))
            group_stories = []
            assembled_length = 0

            while story_index < len(stories):
                story = stories[story_index]
                separator_length = 2 if group_stories else 0
                candidate_length = (
                    assembled_length + separator_length + len(story)
                )

                if candidate_length > len(reference):
                    break

                group_stories.append(story)
                assembled_length = candidate_length
                story_index += 1

                if assembled_length == len(reference):
                    break

            assembled = "\n\n".join(group_stories)

            if assembled != reference:
                raise ValueError(
                    f"Raw stories do not reproduce reference group {path}"
                )

            groups.append(StoryGroup(
                name=path.name,
                stories=tuple(group_stories),
                source_sha256=sha256_file(path),
            ))

        if story_index != len(stories):
            raise ValueError(
                f"Reference corpus consumed {story_index} of "
                f"{len(stories)} complete stories"
            )

        return groups, raw_metadata

    def split_story_groups(
        self,
        groups: list[StoryGroup],
        validation_fraction: float = 0.1,
    ) -> tuple[list[StoryGroup], list[StoryGroup]]:
        """Split historical groups deterministically before tokenizer training."""
        if not 0 < validation_fraction < 1:
            raise ValueError(
                "validation_fraction must be between 0 and 1"
            )

        if len(groups) < 2:
            raise ValueError("At least two story groups are required")

        shuffled = groups.copy()
        random.Random(self.seed).shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * validation_fraction))
        val_count = min(val_count, len(shuffled) - 1)
        return shuffled[val_count:], shuffled[:val_count]

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

    def train_story_tokenizer(
        self,
        train_stories: list[str],
    ) -> Tokenizer:
        """Train a fixed-size byte BPE with a reserved story EOS token."""
        tokenizer = Tokenizer(models.BPE(unk_token=None))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False
        )
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[self.eos_token],
            show_progress=True,
        )

        tokenizer.train_from_iterator(
            train_stories,
            trainer=trainer,
            length=len(train_stories),
        )

        actual_size = tokenizer.get_vocab_size(with_added_tokens=True)

        if actual_size != self.vocab_size:
            raise ValueError(
                f"Tokenizer produced {actual_size} tokens; "
                f"expected {self.vocab_size}. Use a larger training corpus."
            )

        eos_token_id = tokenizer.token_to_id(self.eos_token)

        if eos_token_id is None:
            raise ValueError("Tokenizer did not reserve the EOS token")

        self.tokenizer = tokenizer
        tokenizer.save(str(self.data_dir / "tokenizer.json"))
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

    def encode_stories(self, stories: list[str]) -> np.ndarray:
        """Encode stories and append exactly one EOS token to each."""
        if self.tokenizer is None:
            raise RuntimeError("Train or load the tokenizer first")

        eos_token_id = self.tokenizer.token_to_id(self.eos_token)

        if eos_token_id is None:
            raise ValueError("Tokenizer has no end-of-story token")

        token_ids = []

        for story in stories:
            if self.eos_token in story:
                raise ValueError("Story text contains the reserved EOS token")

            token_ids.extend(self.tokenizer.encode(story).ids)
            token_ids.append(eos_token_id)

        if not token_ids:
            raise ValueError("No tokens produced")

        return np.asarray(token_ids, dtype=np.uint16)

    def prepare_v1_1(
        self,
        raw_file: str | Path,
        reference_corpus_dir: str | Path,
        validation_fraction: float = 0.1,
    ) -> dict:
        """Prepare the deterministic EOS-aware v1.1 training artifacts."""
        groups, raw_metadata = self.load_story_groups(
            raw_file,
            reference_corpus_dir,
        )
        train_groups, val_groups = self.split_story_groups(
            groups,
            validation_fraction,
        )
        train_stories = [
            story for group in train_groups for story in group.stories
        ]
        val_stories = [
            story for group in val_groups for story in group.stories
        ]

        self.train_story_tokenizer(train_stories)
        self.train_tokens = self.encode_stories(train_stories)
        self.val_tokens = self.encode_stories(val_stories)
        self.train_tokens.tofile(self.data_dir / "train.bin")
        self.val_tokens.tofile(self.data_dir / "val.bin")

        eos_token_id = self.tokenizer.token_to_id(self.eos_token)
        group_metadata = {
            group.name: {
                "sha256": group.source_sha256,
                "stories": len(group.stories),
            }
            for group in groups
        }
        artifacts = fingerprint_files(
            self.data_dir,
            ("tokenizer.json", "train.bin", "val.bin"),
        )
        metadata = {
            "format_version": 2,
            "project_version": "1.1",
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "seed": self.seed,
            "eos_token": self.eos_token,
            "eos_token_id": eos_token_id,
            "raw_source": raw_metadata,
            "reference_groups": group_metadata,
            "train_groups": [group.name for group in train_groups],
            "validation_groups": [group.name for group in val_groups],
            "train_stories": len(train_stories),
            "validation_stories": len(val_stories),
            "train_tokens": len(self.train_tokens),
            "validation_tokens": len(self.val_tokens),
            "artifacts": artifacts,
        }
        (self.data_dir / "dataset.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.metadata = metadata
        return metadata

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

    def load_prepared(self, require_eos: bool = False) -> None:
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

        if metadata.get("artifacts"):
            verify_fingerprints(self.data_dir, metadata["artifacts"])

        if self.tokenizer.get_vocab_size() != metadata["vocab_size"]:
            raise ValueError("Tokenizer vocabulary mismatch")

        if self.context_length != metadata["context_length"]:
            raise ValueError("Context length mismatch")

        if self.seed != metadata["seed"]:
            raise ValueError("Prepared dataset seed mismatch")

        if require_eos:
            if metadata.get("eos_token") != self.eos_token:
                raise ValueError("Prepared dataset EOS token mismatch")

            eos_token_id = self.tokenizer.token_to_id(self.eos_token)

            if eos_token_id != metadata.get("eos_token_id"):
                raise ValueError("Prepared dataset EOS token ID mismatch")

        self.metadata = metadata

    def artifact_fingerprints(self) -> dict[str, str]:
        """Return current prepared-data fingerprints for checkpoints."""
        return fingerprint_files(
            self.data_dir,
            ("tokenizer.json", "train.bin", "val.bin", "dataset.json"),
        )

    def fixed_validation_batches(
        self,
        batch_size: int,
        batches: int,
    ):
        """Yield the same validation windows for every evaluation call."""
        if batch_size < 1 or batches < 1:
            raise ValueError("batch_size and batches must be positive")
        if self.val_tokens is None:
            raise RuntimeError("Dataset has not been prepared or loaded")

        window_size = self.context_length + 1

        if len(self.val_tokens) < window_size:
            raise ValueError(
                f"val split has {len(self.val_tokens)} tokens; "
                f"need at least {window_size}"
            )

        rng = np.random.default_rng(self.seed + 1)
        starts = rng.integers(
            0,
            len(self.val_tokens) - window_size + 1,
            size=(batches, batch_size),
        )

        for batch_starts in starts:
            batch = np.stack([
                self.val_tokens[start:start + window_size]
                for start in batch_starts
            ])
            yield mx.array(batch.astype(np.int32))

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


def main():
    """Prepare versioned v1.1 data without touching v1 artifacts."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-file",
        default="data/raw/tinystories-sample.txt",
    )
    parser.add_argument(
        "--reference-corpus",
        default="data/corpus/tinystories",
    )
    parser.add_argument(
        "--data-dir",
        default="data/prepared/tinystories-v1.1",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = LumiDataset(
        data_dir=args.data_dir,
        vocab_size=8192,
        context_length=512,
        seed=args.seed,
    )
    metadata = dataset.prepare_v1_1(
        args.raw_file,
        args.reference_corpus,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
