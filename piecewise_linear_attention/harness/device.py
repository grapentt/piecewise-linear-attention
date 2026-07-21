"""Device selection and seeding — a single source of truth for the harness.

Centralizing these two concerns means every experiment picks a device the same
way and seeds every relevant RNG the same way, which is what makes runs
comparable and reproducible.
"""

import os
import random

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Select a compute device.

    Parameters
    ----------
    prefer:
        ``"auto"`` (default) picks the best available accelerator (CUDA, then
        Apple MPS, then CPU). Any other value is passed straight to
        :class:`torch.device`, so an explicit ``"cpu"``/``"cuda"``/``"mps"``
        forces that device.

    Returns
    -------
    torch.device
        The selected device.
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed all RNGs relevant to a run for reproducibility.

    Seeds Python ``random``, PyTorch (CPU and, if present, CUDA), and sets
    ``PYTHONHASHSEED`` for hash-order stability. NumPy is seeded if importable.

    Parameters
    ----------
    seed:
        The seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:  # NumPy is optional at runtime.
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a torch dep in practice
        pass
