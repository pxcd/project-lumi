"""Pretraining runner for Project Lumi."""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from lumi.dataset import LumiDataset
from lumi.model import LumiConfig, LumiModel
from lumi.train import LumiTrainer


def save_checkpoint(trainer, dataset, directory):
    """Save model, optimizer, and dataset RNG states."""
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
    """Restore trainer and dataset RNG states."""
    directory = Path(directory)

    trainer = LumiTrainer.load_checkpoint(directory)

    rng_state = json.loads(
        (directory / "rng.json").read_text(encoding="utf-8")
    )

    dataset.train_rng.bit_generator.state = rng_state["train"]
    dataset.val_rng.bit_generator.state = rng_state["val"]

    return trainer


def evaluate(trainer, dataset, batch_size, batches):
    """Calculate mean validation loss."""
    losses = []

    for _ in range(batches):
        batch = dataset.get_batch("val", batch_size)
        losses.append(trainer.evaluate(batch))

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
        default="data/prepared/tinystories",
    )

    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/lumi-pretrain",
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
    dataset.load_prepared()

    checkpoint_dir = Path(args.checkpoint_dir)

    if args.resume:
        trainer = load_checkpoint(checkpoint_dir, dataset)
        print(f"Resumed at step {trainer.step}", flush=True)
    else:
        config = LumiConfig(
            vocab_size=8192,
            context_length=512,
            n_layers=8,
            d_model=512,
            n_heads=8,
            d_ff=2048,
        )

        model = LumiModel(config)

        trainer = LumiTrainer(
            model,
            learning_rate=args.learning_rate,
        )

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

            if step % args.save_every == 0:
                save_checkpoint(trainer, dataset, checkpoint_dir)
                print(f"Checkpoint saved: step {step}", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving checkpoint...", flush=True)
        save_checkpoint(trainer, dataset, checkpoint_dir)
        print(f"Saved at step {trainer.step}", flush=True)
        return

    save_checkpoint(trainer, dataset, checkpoint_dir)

    elapsed = time.perf_counter() - start

    print(
        f"\nTraining complete: step={trainer.step} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
