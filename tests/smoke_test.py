"""Project Lumi: automated MLX smoke tests."""

import math
import tempfile
import time
import traceback
from pathlib import Path

import mlx.core as mx

from lumi.model import LumiConfig, LumiModel
from lumi.train import LumiTrainer


CONFIG = LumiConfig(
    vocab_size=256,
    context_length=32,
    n_layers=2,
    d_model=128,
    n_heads=4,
    d_ff=512,
)

results = {}
model = None
trainer = None


def run_test(name, function):
    print(f"\n[RUN] {name}", flush=True)
    start = time.perf_counter()
    try:
        function()
        results[name] = "PASS"
        print(f"[PASS] {name} ({time.perf_counter() - start:.2f}s)", flush=True)
    except Exception:
        results[name] = "FAIL"
        traceback.print_exc()
        print(f"[FAIL] {name}", flush=True)


def test_initialization():
    global model
    model = LumiModel(CONFIG)
    parameters = model.parameters()
    mx.eval(parameters)
    print("Model initialized successfully")


def test_forward():
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    logits = model(tokens)
    mx.eval(logits)

    assert logits.shape == (1, 4, CONFIG.vocab_size)
    assert bool(mx.all(mx.isfinite(logits)).item())
    print("Output shape:", logits.shape)


def test_causality():
    first = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    second = mx.array([[1, 2, 99, 100]], dtype=mx.int32)

    a = model(first)
    b = model(second)
    mx.eval(a, b)

    difference = mx.max(mx.abs(a[:, :2, :] - b[:, :2, :]))
    error = float(difference.item())

    print("Maximum prefix difference:", error)
    assert error < 1e-4, "Future tokens affected earlier outputs"


def test_training():
    global trainer
    trainer = LumiTrainer(model)

    batch = mx.array(
        [[1, 2, 3, 4, 5, 6, 7, 8]],
        dtype=mx.int32,
    )

    before = mx.array(model.embedding.weight)
    mx.eval(before)
    before = mx.array(before)

    loss = trainer.train_step(batch)

    after = model.embedding.weight
    mx.eval(after)

    change = float(mx.max(mx.abs(after - before)).item())

    assert math.isfinite(loss)
    assert change > 0, "Optimizer did not update weights"
    assert trainer.step == 1

    print(f"Loss: {loss:.4f}")
    print(f"Maximum weight change: {change:.8f}")


def test_checkpoint():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        trainer.save_checkpoint(path)

        restored = LumiTrainer.load_checkpoint(path)
        assert restored.step == trainer.step

        tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

        original = trainer.model(tokens)
        reloaded = restored.model(tokens)
        mx.eval(original, reloaded)

        error = float(mx.max(mx.abs(original - reloaded)).item())
        print("Maximum checkpoint output difference:", error)

        assert error < 1e-5, "Restored weights differ"


def main():
    print("PROJECT LUMI — GPU SMOKE TEST")
    print("=" * 40)

    run_test("Model initialization", test_initialization)

    if results["Model initialization"] == "PASS":
        run_test("Forward pass", test_forward)

        if results["Forward pass"] == "PASS":
            run_test("Causal attention", test_causality)
            run_test("Training step", test_training)

            if results["Training step"] == "PASS":
                run_test("Checkpoint restoration", test_checkpoint)

    print("\n" + "=" * 40)
    print("FINAL RESULTS")
    for name, status in results.items():
        print(f"{status:4} | {name}")

    if any(status == "FAIL" for status in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
