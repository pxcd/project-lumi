"""Focused tests for Lumi v1.1 EOS and reproducibility behavior."""

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from lumi.dataset import END_OF_STORY_TOKEN, LumiDataset
from lumi.generate import generate
from lumi.model import LumiConfig
from lumi.pretrain import (
    build_trainer,
    evaluate,
    load_checkpoint,
    save_checkpoint,
)


DELIMITER = "<|endoftext|>"


class PreparedFixture:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.raw = root / "raw.txt"
        self.corpus = root / "corpus"
        self.prepared = root / "prepared"
        self.corpus.mkdir()
        self.stories = [
            "Mira found a small red kite beside the old oak tree.",
            "Tom carried the kite home and repaired its torn blue tail.",
            "The next morning, Mira and Tom flew it above the quiet park.",
            "When the wind stopped, they shared warm bread with their dog.",
        ]
        self.raw.write_text(
            self.stories[0]
            + DELIMITER
            + DELIMITER
            + self.stories[1]
            + DELIMITER
            + self.stories[2]
            + DELIMITER
            + self.stories[3]
            + DELIMITER
            + "This final record is deliberately truncated",
            encoding="utf-8",
        )
        (self.corpus / "stories_000.txt").write_text(
            "\n\n".join(self.stories[:2]),
            encoding="utf-8",
        )
        (self.corpus / "stories_001.txt").write_text(
            "\n\n".join(self.stories[2:]),
            encoding="utf-8",
        )
        self.dataset = LumiDataset(
            data_dir=self.prepared,
            vocab_size=260,
            context_length=4,
            seed=42,
        )
        self.metadata = self.dataset.prepare_v1_1(
            self.raw,
            self.corpus,
            validation_fraction=0.5,
        )


class DatasetTests(unittest.TestCase):
    def test_preparation_preserves_exact_story_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedFixture(Path(directory))
            metadata = fixture.metadata
            eos_id = metadata["eos_token_id"]

            self.assertEqual(metadata["raw_source"]["complete_stories"], 4)
            self.assertEqual(metadata["raw_source"]["empty_records"], 1)
            self.assertTrue(
                metadata["raw_source"]["dropped_trailing_fragment"]
            )
            self.assertEqual(metadata["train_stories"], 2)
            self.assertEqual(metadata["validation_stories"], 2)
            self.assertEqual(
                int(np.count_nonzero(fixture.dataset.train_tokens == eos_id)),
                2,
            )
            self.assertEqual(
                int(np.count_nonzero(fixture.dataset.val_tokens == eos_id)),
                2,
            )

    def test_tokenizer_reserves_single_eos_inside_fixed_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedFixture(Path(directory))
            tokenizer = fixture.dataset.tokenizer
            eos_id = tokenizer.token_to_id(END_OF_STORY_TOKEN)

            self.assertIsNotNone(eos_id)
            self.assertEqual(tokenizer.get_vocab_size(), 260)
            self.assertEqual(
                tokenizer.encode(END_OF_STORY_TOKEN).ids,
                [eos_id],
            )

            reloaded = LumiDataset(
                data_dir=fixture.prepared,
                vocab_size=260,
                context_length=4,
                seed=42,
            )
            reloaded.load_prepared(require_eos=True)
            self.assertEqual(reloaded.metadata, fixture.metadata)

    def test_validation_windows_are_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedFixture(Path(directory))

            class SumTrainer:
                @staticmethod
                def evaluate(batch):
                    return float(np.asarray(batch).sum())

            first = evaluate(SumTrainer(), fixture.dataset, 2, 3)
            fixture.dataset.get_batch("val", 2)
            second = evaluate(SumTrainer(), fixture.dataset, 2, 3)
            self.assertEqual(first, second)


class TrainingTests(unittest.TestCase):
    @staticmethod
    def tiny_config(vocab_size=260):
        return LumiConfig(
            vocab_size=vocab_size,
            context_length=4,
            n_layers=1,
            d_model=32,
            n_heads=4,
            d_ff=64,
        )

    @staticmethod
    def flattened_parameters(trainer):
        parameters = tree_flatten(trainer.model.parameters())
        mx.eval([value for _, value in parameters])
        return {
            name: np.asarray(value).copy()
            for name, value in parameters
        }

    def test_initialization_is_deterministic(self):
        config = self.tiny_config()
        first = build_trainer(17, config=config)
        first_parameters = self.flattened_parameters(first)
        second = build_trainer(17, config=config)
        second_parameters = self.flattened_parameters(second)
        third = build_trainer(18, config=config)
        third_parameters = self.flattened_parameters(third)

        for name in first_parameters:
            np.testing.assert_array_equal(
                first_parameters[name],
                second_parameters[name],
            )

        self.assertTrue(any(
            not np.array_equal(first_parameters[name], third_parameters[name])
            for name in first_parameters
        ))

    def test_checkpoint_recovers_state_and_verifies_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = PreparedFixture(root / "fixture")
            trainer = build_trainer(42, config=self.tiny_config())
            trainer.train_step(fixture.dataset.get_batch("train", 1))
            checkpoint = root / "checkpoints" / "latest"
            state = {
                "seed": 42,
                "best_validation_loss": 1.25,
                "last_validation_loss": 1.5,
                "run_config": {"seed": 42},
            }
            save_checkpoint(
                trainer,
                fixture.dataset,
                checkpoint,
                state,
            )
            restored, restored_state = load_checkpoint(
                checkpoint,
                fixture.dataset,
            )

            self.assertEqual(restored.step, trainer.step)
            self.assertEqual(restored_state, state)
            self.assertTrue((checkpoint / "manifest.json").is_file())

            inputs = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
            original_logits = trainer.model(inputs)
            restored_logits = restored.model(inputs)
            mx.eval(original_logits, restored_logits)
            np.testing.assert_allclose(
                np.asarray(original_logits),
                np.asarray(restored_logits),
                rtol=0,
                atol=1e-6,
            )

            next_batch = fixture.dataset.get_batch("train", 1)
            trainer.train_step(next_batch)
            restored.train_step(next_batch)
            original_logits = trainer.model(inputs)
            restored_logits = restored.model(inputs)
            mx.eval(original_logits, restored_logits)
            np.testing.assert_allclose(
                np.asarray(original_logits),
                np.asarray(restored_logits),
                rtol=0,
                atol=1e-6,
            )

            with (fixture.prepared / "val.bin").open("ab") as file:
                file.write(b"\x00\x00")

            with self.assertRaisesRegex(ValueError, "fingerprints"):
                load_checkpoint(checkpoint, fixture.dataset)


class GenerationTests(unittest.TestCase):
    def test_generation_stops_on_eos(self):
        class Encoding:
            ids = [1]

        class Tokenizer:
            @staticmethod
            def encode(_text):
                return Encoding()

            @staticmethod
            def decode(ids, skip_special_tokens):
                self.assertTrue(skip_special_tokens)
                return "A" if ids == [2, 3] else "unexpected"

        class Config:
            context_length = 8

        class Model:
            config = Config()

            @staticmethod
            def eval():
                return None

            def __call__(self, inputs):
                return mx.zeros((1, inputs.shape[1], 4))

        sampled = iter((2, 3, 2))
        result = generate(
            Model(),
            Tokenizer(),
            "prompt",
            max_tokens=10,
            temperature=1.0,
            seed=42,
            eos_token_id=3,
            sample_token=lambda _logits: next(sampled),
        )

        self.assertEqual(result.continuation, "A")
        self.assertEqual(result.generated_tokens, 2)
        self.assertEqual(result.finish_reason, "eos")
        self.assertEqual(next(sampled), 2)


if __name__ == "__main__":
    unittest.main()
