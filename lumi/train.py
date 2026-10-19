
"""Training utilities for Lumi's decoder-only Transformer."""

from pathlib import Path
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from lumi.model import LumiConfig, LumiModel


def next_token_loss(
    model: LumiModel,
    batch: mx.array,
) -> mx.array:
    """Predict each token from all preceding tokens."""
    if batch.ndim != 2 or batch.shape[1] < 2:
        raise ValueError("Batch must have shape [batch, sequence >= 2]")

    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model(inputs)

    return nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )


class LumiTrainer:
    def __init__(
        self,
        model: LumiModel,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.optimizer = optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        self.max_grad_norm = max_grad_norm
        self.loss_and_grad = nn.value_and_grad(
            model, next_token_loss
        )
        self.step = 0

    def train_step(self, batch: mx.array) -> float:
        self.model.train()

        loss, grads = self.loss_and_grad(self.model, batch)

        grads, _ = optim.clip_grad_norm(
            grads, self.max_grad_norm
        )

        self.optimizer.update(self.model, grads)

        # Force evaluation of lazy MLX operations.
        mx.eval(
            loss,
            self.model.parameters(),
            self.optimizer.state,
        )

        self.step += 1
        return float(loss.item())

    def evaluate(self, batch: mx.array) -> float:
        self.model.eval()
        loss = next_token_loss(self.model, batch)
        mx.eval(loss)
        return float(loss.item())

    def fit(
        self,
        batches,
        steps: int,
        log_every: int = 10,
    ) -> None:
        """Train on batches supplied by an iterable."""
        if steps < 1:
            raise ValueError("steps must be positive")

        iterator = iter(batches)

        for _ in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(batches)
                batch = next(iterator)

            loss = self.train_step(batch)

            if self.step % log_every == 0 or self.step == 1:
                print(f"step={self.step} loss={loss:.4f}")

    def save_checkpoint(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.model.save_weights(
            str(directory / "model.safetensors")
        )

        metadata = {
            "step": self.step,
            "config": vars(self.model.config),
            "optimizer": "AdamW",
        }

        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_checkpoint(
        cls,
        directory: str | Path,
        learning_rate: float = 3e-4,
    ) -> "LumiTrainer":
        directory = Path(directory)

        metadata = json.loads(
            (directory / "metadata.json").read_text(
                encoding="utf-8"
            )
        )

        config = LumiConfig(**metadata["config"])
        model = LumiModel(config)

        model.load_weights(
            str(directory / "model.safetensors")
        )

        trainer = cls(model, learning_rate=learning_rate)
        trainer.step = metadata["step"]

        return trainer
