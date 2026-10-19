"""Small helpers for reproducible artifact fingerprints."""

from hashlib import sha256
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def fingerprint_files(
    directory: str | Path,
    names: tuple[str, ...],
) -> dict[str, str]:
    """Fingerprint named files relative to a directory."""
    directory = Path(directory)
    return {
        name: sha256_file(directory / name)
        for name in names
    }


def verify_fingerprints(
    directory: str | Path,
    expected: dict[str, str],
) -> None:
    """Raise when a file is missing or differs from its recorded digest."""
    directory = Path(directory)

    for name, expected_digest in expected.items():
        path = directory / name

        if not path.is_file():
            raise ValueError(f"Missing fingerprinted artifact: {path}")

        actual_digest = sha256_file(path)

        if actual_digest != expected_digest:
            raise ValueError(
                f"Artifact fingerprint mismatch for {path}: "
                f"expected {expected_digest}, got {actual_digest}"
            )
