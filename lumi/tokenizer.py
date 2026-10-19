
"""Byte-level BPE tokenizer for Project Lumi."""

from collections import Counter
from pathlib import Path
import json


class LumiTokenizer:
    def __init__(self):
        self.vocab: dict[int, bytes] = {
            i: bytes([i]) for i in range(256)
        }
        self.merges: dict[tuple[int, int], int] = {}

    @staticmethod
    def count_pairs(token_ids: list[int]) -> dict[tuple[int, int], int]:
        return dict(Counter(zip(token_ids, token_ids[1:])))

    @staticmethod
    def merge_pair(
        token_ids: list[int],
        pair: tuple[int, int],
        new_token_id: int,
    ) -> list[int]:
        result = []
        i = 0

        while i < len(token_ids):
            if (
                i + 1 < len(token_ids)
                and (token_ids[i], token_ids[i + 1]) == pair
            ):
                result.append(new_token_id)
                i += 2
            else:
                result.append(token_ids[i])
                i += 1

        return result

    def register_merge(self, pair: tuple[int, int]) -> int:
        if pair in self.merges:
            raise ValueError(f"Duplicate merge: {pair}")

        if pair[0] not in self.vocab or pair[1] not in self.vocab:
            raise ValueError(f"Unknown token in pair: {pair}")

        token_id = len(self.vocab)
        self.vocab[token_id] = (
            self.vocab[pair[0]] + self.vocab[pair[1]]
        )
        self.merges[pair] = token_id
        return token_id

    def train(self, text: str, vocab_size: int = 8192) -> None:
        if vocab_size < 256:
            raise ValueError("Vocabulary size must be at least 256")

        # Each training run starts from the base byte vocabulary.
        self.__init__()
        token_ids = list(text.encode("utf-8"))

        while len(self.vocab) < vocab_size:
            counts = self.count_pairs(token_ids)

            if not counts:
                break

            pair = max(counts, key=counts.get)
            token_id = self.register_merge(pair)
            token_ids = self.merge_pair(token_ids, pair, token_id)

    def encode(self, text: str) -> list[int]:
        token_ids = list(text.encode("utf-8"))

        # Apply learned merges in training order.
        for pair, token_id in self.merges.items():
            token_ids = self.merge_pair(token_ids, pair, token_id)

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        data = b"".join(self.vocab[i] for i in token_ids)
        return data.decode("utf-8")

    def save(self, path: str | Path) -> None:
        data = {
            "version": 1,
            "merges": [
                [a, b] for a, b in self.merges
            ],
        }

        Path(path).write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "LumiTokenizer":
        data = json.loads(
            Path(path).read_text(encoding="utf-8")
        )

        if data.get("version") != 1:
            raise ValueError("Unsupported tokenizer version")

        tokenizer = cls()

        for pair in data["merges"]:
            tokenizer.register_merge(tuple(pair))

        return tokenizer
