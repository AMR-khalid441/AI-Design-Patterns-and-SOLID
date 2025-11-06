"""Shared test helpers (stubs).

Functions here should assist tests without coupling violation and solution code.
"""

def set_deterministic_seed(seed: int) -> None:
    """Stub: set random seeds for reproducible tests."""
    pass

def make_tmp_output_dir() -> str:
    """Stub: return a temporary output directory path for tests."""
    return "./outputs"


