"""ListOps dataset for LRA benchmarks.

ListOps is a diagnostic task that requires parsing hierarchical nested
expressions like: [MAX 2 9 [MIN 4 7] 0].

Dataset format: TSV with columns (Source, Target)
- Source: ListOps expression as string
- Target: Result value (0-9)
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
import urllib.request
from pathlib import Path


class ListOpsDataset(Dataset):
    """ListOps dataset.

    Character-level tokenization for parsing nested list operations.
    """

    # Character-level vocabulary (optimized for ListOps expressions)
    VOCAB = {
        '<PAD>': 0,   # Padding token
        '<UNK>': 1,   # Unknown token
        '[': 2,       # List open
        ']': 3,       # List close
        'M': 4,       # For MAX, MIN, MED
        'A': 5,       # For MAX, MED
        'X': 6,       # For MAX
        'I': 7,       # For MIN, MED
        'N': 8,       # For MIN
        'E': 9,       # For MED
        'D': 10,      # For MED
        'S': 11,      # For SM (sum modulo)
        ' ': 12,      # Space
        '0': 13,      # Digits
        '1': 14,
        '2': 15,
        '3': 16,
        '4': 17,
        '5': 18,
        '6': 19,
        '7': 20,
        '8': 21,
        '9': 22,
    }

    URLS = {
        "train": "https://storage.googleapis.com/long-range-arena/lra_release/listops-1000/basic_train.tsv",
        "val": "https://storage.googleapis.com/long-range-arena/lra_release/listops-1000/basic_val.tsv",
        "test": "https://storage.googleapis.com/long-range-arena/lra_release/listops-1000/basic_test.tsv",
    }

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_length: int = 2000,
        download: bool = True,
    ):
        """Initialize ListOps dataset.

        Args:
            data_dir: Directory containing listops data
            split: "train", "val", or "test"
            max_length: Maximum sequence length
            download: Whether to download data if not found
        """
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be train, val, or test.")

        self.data_dir = Path(data_dir)
        self.split = split
        self.max_length = max_length

        # Download if needed
        if download:
            self._download_if_needed()

        # Load data
        self.examples = self._load_data()

    def _download_if_needed(self):
        """Generate ListOps data if not present (GCS bucket has restricted access)."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        filepath = self.data_dir / f"basic_{self.split}.tsv"

        if not filepath.exists():
            # The GCS bucket (gs://long-range-arena/lra_release) currently has
            # restricted access (403 Forbidden), so we generate locally using the
            # official LRA generation code. This ensures reproducibility.
            print(f"ListOps dataset not found. Generating locally...")

            # Generate all splits at once (only do this from the train split loader)
            if self.split == "train":
                self._generate_locally()
            else:
                # For val/test, wait briefly for train to generate all splits
                import time
                for _ in range(60):  # Wait up to 60 seconds
                    if filepath.exists():
                        break
                    time.sleep(1)

                if not filepath.exists():
                    # If still not found, generate ourselves
                    self._generate_locally()

    def _load_data(self) -> list:
        """Load data from TSV file."""
        filepath = self.data_dir / f"basic_{self.split}.tsv"

        if not filepath.exists():
            raise FileNotFoundError(
                f"ListOps data not found at {filepath}. "
                f"Run with download=True to download automatically."
            )

        examples = []
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    source, target = parts
                    try:
                        examples.append((source, int(target)))
                    except ValueError:
                        print(f"Warning: Invalid target on line {line_num}: {target}")
                        continue

        print(f"✓ Loaded {len(examples)} {self.split} examples")
        return examples

    def _tokenize(self, text: str) -> list:
        """Tokenize text to character IDs.

        Args:
            text: Input string

        Returns:
            List of token IDs
        """
        tokens = []
        for char in text:
            tokens.append(self.VOCAB.get(char, self.VOCAB['<UNK>']))
        return tokens

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        source, target = self.examples[idx]

        # Tokenize
        tokens = self._tokenize(source)

        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        # Create padding mask (True for valid tokens, False for padding)
        padding_mask = [True] * len(tokens)

        # Pad to max_length
        while len(tokens) < self.max_length:
            tokens.append(self.VOCAB['<PAD>'])
            padding_mask.append(False)

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
            'labels': torch.tensor(target, dtype=torch.long),
        }

    def _generate_locally(self):
        """Generate dataset locally using generate_listops.py."""
        import subprocess
        import sys

        print(f"  Generating all splits (train, val, test)...")
        print(f"  This may take a few minutes for 100K samples...")

        # Get the generator script path
        generator_script = Path(__file__).parent / "generate_listops.py"

        # Run generator
        cmd = [
            sys.executable,
            str(generator_script),
            "--output-dir", str(self.data_dir),
            "--seed", "42",
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            # Print the generator output
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        print(f"  {line}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Generation failed")
            if e.stderr:
                print(f"  Error: {e.stderr}")
            raise RuntimeError(
                f"Failed to generate ListOps dataset. "
                f"Please run manually: python {generator_script} --output-dir {self.data_dir}"
            )

    @staticmethod
    def get_vocab_size() -> int:
        """Get vocabulary size."""
        return len(ListOpsDataset.VOCAB)


def get_listops_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    max_length: int = 2000,
    download: bool = True,
    num_workers: int = 0,  # Default to 0 for Mac compatibility
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get ListOps dataloaders.

    Args:
        data_dir: Directory containing listops data
        batch_size: Batch size
        max_length: Maximum sequence length
        download: Whether to download data if not found
        num_workers: Number of data loading workers (0 for main thread)

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = ListOpsDataset(data_dir, "train", max_length, download)
    val_dataset = ListOpsDataset(data_dir, "val", max_length, download)
    test_dataset = ListOpsDataset(data_dir, "test", max_length, download)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
