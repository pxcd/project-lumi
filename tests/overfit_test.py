
"""Verify that Lumi can memorize a tiny training sequence."""

import math

import mlx.core as mx

from lumi.model import LumiConfig, LumiModel
from lumi.train import LumiTrainer


def main():
    mx.random.seed(42)

    config = LumiConfig(
        vocab_size=256,
        context_length=32,
        n_layers=2,
        d_model=128,
        n_heads=4,
        d_ff=512,
    )

    model = LumiModel(config)
    trainer = LumiTrainer(
        model,
        learning_rate=1e-3,
        weight_decay=0.0,
    )

    text = "Hello, Lumi! "
    tokens = list(text.encode("utf-8"))
    batch = mx.array([tokens], dtype=mx.int32)

    # Materialize initial weights before taking measurements.
    mx.eval(model.parameters())

    initial_loss = trainer.evaluate(batch)
    print(f"Initial loss: {initial_loss:.4f}", flush=True)

    for step in range(1, 201):
        loss = trainer.train_step(batch)

        if step % 20 == 0:
            print(f"Step {step:3d}: loss={loss:.4f}", flush=True)

    final_loss = trainer.evaluate(batch)
    print(f"Final loss:   {final_loss:.4f}", flush=True)

    assert math.isfinite(final_loss)
    assert final_loss < initial_loss * 0.2, (
        "Model failed to substantially reduce training loss"
    )

    print("PASS: Lumi learned the training sequence!")


if __name__ == "__main__":
    main()
