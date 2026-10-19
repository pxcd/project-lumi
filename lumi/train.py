
"""Training utilities for Lumi's decoder-only Transformer."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from lumi.model import LumiConfig, LumiModel


def next_token_loss(
    model: LumiModel,
    batch: mx.array,
) -> mx.array:
    """Calculate next-token prediction loss."""
    if batch.ndim != 2 or batch.shape[1] < 2:
        raise ValueError(
            "Batch must have shape [batch, sequence >= 2]"
        )

    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model(inputs)

    return nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )


def restore_optimizer_state(target, source) -> None:
    """Restore saved values while preserving empty optimizer branches."""
    if isinstance(target, dict):
        for key, value in source.items():
            if isinstance(value, (dict, list)):
                restore_optimizer_state(target[key], value)
            else:
                target[key] = value

    elif isinstance(target, list):
        for index, value in enumerate(source):
            if isinstance(value, (dict, list)):
                restore_optimizer_state(target[index], value)
            else:
                target[index] = value


class LumiTrainer:
    def __init__(
        self,
        model: LumiModel,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
    ):
        self.model = model

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        self.loss_and_grad = nn.value_and_grad(
            model, next_token_loss
        )

        self.step = 0

    def train_step(self, batch: mx.array) -> float:
        """Perform one optimization step."""
        self.model.train()

        loss, grads = self.loss_and_grad(
            self.model, batch
        )

        grads, _ = optim.clip_grad_norm(
            grads, self.max_grad_norm
        )

        self.optimizer.update(
            self.model, grads
        )

        mx.eval(
            loss,
            self.model.parameters(),
            self.optimizer.state,
        )

        self.step += 1

        return float(loss.item())

    def evaluate(self, batch: mx.array) -> float:
        """Evaluate a batch without updating weights."""
        self.model.eval()

        loss = next_token_loss(
            self.model, batch
        )

        mx.eval(loss)

        return float(loss.item())

    def fit(
        self,
        batches,
        steps: int,
        log_every: int = 10,
    ) -> None:
        """Train on an iterable of token batches."""
        if steps < 1:
            raise ValueError("steps must be positive")

        if log_every < 1:
            raise ValueError("log_every must be positive")

        iterator = iter(batches)

        for _ in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(batches)
                batch = next(iterator)

            loss = self.train_step(batch)

            if (
                self.step % log_every == 0
                or self.step == 1
            ):
                print(
                    f"step={self.step} loss={loss:.4f}",
                    flush=True,
                )

    def save_checkpoint(
        self,
        directory: str | Path,
    ) -> None:
        """Save weights, optimizer state, and metadata."""
        directory = Path(directory)
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        mx.eval(
            self.model.parameters(),
            self.optimizer.state,
        )

        self.model.save_weights(
            str(directory / "model.safetensors")
        )

        optimizer_state = dict(
            tree_flatten(self.optimizer.state)
        )

        mx.save_safetensors(
            str(directory / "optimizer.safetensors"),
            optimizer_state,
        )

        metadata = {
            "version": 2,
            "step": self.step,
            "config": vars(self.model.config),
            "optimizer": {
                "type": "AdamW",
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "max_grad_norm": self.max_grad_norm,
            },
        }

        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_checkpoint(
        cls,
        directory: str | Path,
    ) -> "LumiTrainer":
        """Restore model weights and AdamW optimizer state."""
        directory = Path(directory)

        metadata = json.loads(
            (directory / "metadata.json").read_text(
                encoding="utf-8"
            )
        )

        if metadata.get("version") != 2:
            raise ValueError(
                "Expected a version 2 checkpoint"
            )

        config = LumiConfig(
            **metadata["config"]
        )

        model = LumiModel(config)

        model.load_weights(
            str(directory / "model.safetensors")
        )

        optimizer_config = metadata["optimizer"]

        if optimizer_config["type"] != "AdamW":
            raise ValueError(
                "Unsupported optimizer type"
            )

        trainer = cls(
            model,
            learning_rate=optimizer_config[
                "learning_rate"
            ],
            weight_decay=optimizer_config[
                "weight_decay"
            ],
            max_grad_norm=optimizer_config[
                "max_grad_norm"
            ],
        )

        # Build the complete optimizer structure,
        # including empty entries for modules like RoPE.
        trainer.optimizer.init(
            model.trainable_parameters()
        )

        saved_state = mx.load(
            str(directory / "optimizer.safetensors")
        )

        restored_state = tree_unflatten(
            list(saved_state.items())
        )

        # Merge rather than replace. Flattened checkpoints
        # omit empty dictionaries such as rope: {}.
        restore_optimizer_state(
            trainer.optimizer.state,
            restored_state,
        )

        trainer.step = metadata["step"]

        mx.eval(
            trainer.model.parameters(),
            trainer.optimizer.state,
        )

        return trainer
