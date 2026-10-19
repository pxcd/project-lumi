"""Pretraining runner for Project Lumi."""

import argparse
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from lumi.artifacts import fingerprint_files, verify_fingerprints
from lumi.dataset import LumiDataset
from lumi.model import LumiConfig, LumiModel
from lumi.train import LumiTrainer


CHECKPOINT_FILES = (
    "model.safetensors",
    "optimizer.safetensors",
    "metadata.json",
    "rng.json",
    "training_state.json",
)


def build_trainer(seed, learning_rate=3e-4, config=None):
    """Build a deterministically initialized Lumi trainer."""
    mx.random.seed(seed)
    config = config or LumiConfig(
        vocab_size=8192,
        context_length=512,
        n_layers=8,
        d_model=512,
        n_heads=8,
        d_ff=2048,
    )
    return LumiTrainer(
        LumiModel(config),
        learning_rate=learning_rate,
    )


def save_checkpoint(trainer, dataset, directory, training_state):
    """Atomically save a fingerprinted training checkpoint."""
    directory = Path(directory)
    temporary = directory.with_name(directory.name + ".tmp")

    if temporary.exists():
        shutil.rmtree(temporary)

    trainer.save_checkpoint(temporary)

    rng_state = {
        "train": dataset.train_rng.bit_generator.state,
        "val": dataset.val_rng.bit_generator.state,
    }

    (temporary / "rng.json").write_text(
        json.dumps(rng_state, indent=2),
        encoding="utf-8",
    )

    (temporary / "training_state.json").write_text(
        json.dumps(training_state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "format_version": 1,
        "checkpoint_artifacts": fingerprint_files(
            temporary,
            CHECKPOINT_FILES,
        ),
        "dataset_artifacts": dataset.artifact_fingerprints(),
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    backup = directory.with_name(directory.name + ".bak")

    if backup.exists():
        shutil.rmtree(backup)

    if directory.exists():
        directory.rename(backup)

    try:
        temporary.rename(directory)
    except Exception:
        if backup.exists() and not directory.exists():
            backup.rename(directory)
        raise

    if backup.exists():
        shutil.rmtree(backup)


def load_checkpoint(directory, dataset):
    """Restore a checkpoint only when all fingerprints still match."""
    directory = Path(directory)

    manifest = json.loads(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    verify_fingerprints(
        directory,
        manifest["checkpoint_artifacts"],
    )

    current_dataset_artifacts = dataset.artifact_fingerprints()

    if current_dataset_artifacts != manifest["dataset_artifacts"]:
        raise ValueError("Checkpoint dataset fingerprints do not match")

    trainer = LumiTrainer.load_checkpoint(directory)

    if trainer.model.config.vocab_size != dataset.vocab_size:
        raise ValueError("Checkpoint vocabulary does not match dataset")

    if trainer.model.config.context_length != dataset.context_length:
        raise ValueError("Checkpoint context length does not match dataset")

    rng_state = json.loads(
        (directory / "rng.json").read_text(encoding="utf-8")
    )

    dataset.train_rng.bit_generator.state = rng_state["train"]
    dataset.val_rng.bit_generator.state = rng_state["val"]

    training_state = json.loads(
        (directory / "training_state.json").read_text(encoding="utf-8")
    )

    return trainer, training_state


def evaluate(trainer, dataset, batch_size, batches):
    """Calculate mean loss on fixed validation windows."""
    losses = [
        trainer.evaluate(batch)
        for batch in dataset.fixed_validation_batches(batch_size, batches)
    ]

    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--steps", type=int, default=11000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument(
        "--data-dir",
        default="data/prepared/tinystories-v1.1",
    )

    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/lumi-v1.1",
    )

    args = parser.parse_args()

    for name in (
        "steps", "batch_size", "log_every",
        "eval_every", "eval_batches", "save_every",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    dataset = LumiDataset(
        data_dir=args.data_dir,
        vocab_size=8192,
        context_length=512,
        seed=args.seed,
    )
    dataset.load_prepared(require_eos=True)

    checkpoint_root = Path(args.checkpoint_dir)
    latest_checkpoint = checkpoint_root / "latest"
    best_checkpoint = checkpoint_root / "best"
    run_config = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "log_every": args.log_every,
        "eval_every": args.eval_every,
        "eval_batches": args.eval_batches,
        "save_every": args.save_every,
        "seed": args.seed,
    }

    if args.resume:
        trainer, training_state = load_checkpoint(
            latest_checkpoint,
            dataset,
        )

        if training_state["seed"] != args.seed:
            raise ValueError("Checkpoint seed does not match --seed")

        if training_state["run_config"] != run_config:
            raise ValueError("Checkpoint training configuration mismatch")

        print(f"Resumed at step {trainer.step}", flush=True)
    else:
        trainer = build_trainer(args.seed, args.learning_rate)
        training_state = {
            "seed": args.seed,
            "best_validation_loss": None,
            "last_validation_loss": None,
            "run_config": run_config,
        }

    if trainer.step >= args.steps:
        print("Target step already reached.")
        return

    print(
        f"Training steps {trainer.step + 1}–{args.steps}",
        flush=True,
    )

    start = time.perf_counter()
    interval_start = start

    try:
        while trainer.step < args.steps:
            batch = dataset.get_batch("train", args.batch_size)
            loss = trainer.train_step(batch)
            step = trainer.step

            if step % args.log_every == 0 or step == 1:
                elapsed = time.perf_counter() - interval_start
                print(
                    f"step={step} train_loss={loss:.4f} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
                interval_start = time.perf_counter()

            if step % args.eval_every == 0:
                val_loss = evaluate(
                    trainer,
                    dataset,
                    args.batch_size,
                    args.eval_batches,
                )
                print(
                    f"step={step} val_loss={val_loss:.4f}",
                    flush=True,
                )
                training_state["last_validation_loss"] = val_loss
                best_loss = training_state["best_validation_loss"]

                if best_loss is None or val_loss < best_loss:
                    training_state["best_validation_loss"] = val_loss
                    save_checkpoint(
                        trainer,
                        dataset,
                        best_checkpoint,
                        training_state,
                    )
                    print(
                        f"Best checkpoint saved: step {step}",
                        flush=True,
                    )

            if step % args.save_every == 0:
                save_checkpoint(
                    trainer,
                    dataset,
                    latest_checkpoint,
                    training_state,
                )
                print(f"Checkpoint saved: step {step}", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving checkpoint...", flush=True)
        save_checkpoint(
            trainer,
            dataset,
            latest_checkpoint,
            training_state,
        )
        print(f"Saved at step {trainer.step}", flush=True)
        return

    save_checkpoint(
        trainer,
        dataset,
        latest_checkpoint,
        training_state,
    )

    elapsed = time.perf_counter() - start

    print(
        f"\nTraining complete: step={trainer.step} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
