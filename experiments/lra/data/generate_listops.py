"""Generate ListOps dataset locally.

Based on: lra_benchmarks/data/listops.py
This generates the ListOps dataset when the GCS bucket is not accessible.
"""
import csv
import random
import argparse
from pathlib import Path
import numpy as np


# Operators and values
MIN = '[MIN'
MAX = '[MAX'
MED = '[MED'
SUM_MOD = '[SM'
END = ']'

OPERATORS = [MIN, MAX, MED, SUM_MOD]
VALUES = list(range(10))
VALUE_P = 0.25


def generate_tree(depth: int, max_depth: int, max_args: int):
    """Generate tree-like equations.

    Args:
        depth: Current depth of the node
        max_depth: Maximum depth of the tree
        max_args: Maximum number of arguments per operator

    Returns:
        Tuple of (tree_structure, length)
    """
    if depth < max_depth:
        r = random.random()
    else:
        r = 1.0

    if r > VALUE_P:
        # Generate a value
        value = random.choice(VALUES)
        return value, 1
    else:
        # Generate an operator with arguments
        length = 2
        num_values = random.randint(2, max_args)
        values = []
        for _ in range(num_values):
            sub_t, sub_l = generate_tree(depth + 1, max_depth, max_args)
            values.append(sub_t)
            length += sub_l

        op = random.choice(OPERATORS)
        t = (op, values[0])
        for value in values[1:]:
            t = (t, value)
        t = (t, END)

    return t, length


def to_string(t) -> str:
    """Convert tree to string representation."""
    if isinstance(t, str):
        return t
    elif isinstance(t, int):
        return str(t)
    else:
        return to_string(t[0]) + ' ' + to_string(t[1])


def to_value(t) -> int:
    """Compute the output of equation t.

    Args:
        t: Tree structure representing the equation

    Returns:
        Result of the equation
    """
    if not isinstance(t, tuple):
        return t

    l = to_value(t[0])
    r = to_value(t[1])

    if l in OPERATORS:
        # Create an unsaturated function
        return (l, [r])
    elif r == END:
        # l must be an unsaturated function
        if l[0] == MIN:
            return min(l[1])
        elif l[0] == MAX:
            return max(l[1])
        elif l[0] == MED:
            return int(np.median(l[1]))
        elif l[0] == SUM_MOD:
            return int(np.sum(l[1]) % 10)
    elif isinstance(l, tuple):
        # Unsaturated function + argument
        return (l[0], l[1] + [r])

    raise ValueError(f"Invalid tree structure: l={l}, r={r}")


def generate_samples(
    num_samples: int,
    max_depth: int,
    max_args: int,
    min_length: int,
    max_length: int,
) -> list:
    """Generate dataset samples.

    Args:
        num_samples: Number of samples to generate
        max_depth: Maximum tree depth
        max_args: Maximum arguments per operator
        min_length: Minimum sequence length
        max_length: Maximum sequence length

    Returns:
        List of (source_string, target_value) tuples
    """
    samples = []

    for _ in range(num_samples):
        # Keep generating until we get a valid length
        while True:
            tree, length = generate_tree(0, max_depth, max_args)
            if min_length <= length <= max_length:
                break

        # Convert to string and compute value
        source = to_string(tree)
        target = to_value(tree)

        samples.append((source, target))

    return samples


def write_to_file(samples: list, filepath: Path):
    """Write samples to TSV file.

    Args:
        samples: List of (source, target) tuples
        filepath: Output file path
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for source, target in samples:
            writer.writerow([source, target])

    print(f"✓ Wrote {len(samples)} samples to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate ListOps dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/lra/listops",
        help="Output directory",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=96000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=2000,
        help="Number of validation samples",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=2000,
        help="Number of test samples",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum tree depth",
    )
    parser.add_argument(
        "--max-args",
        type=int,
        default=10,
        help="Maximum arguments per operator",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=500,
        help="Minimum sequence length",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2000,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating ListOps dataset with seed={args.seed}")
    print(f"  Train samples: {args.num_train}")
    print(f"  Val samples: {args.num_val}")
    print(f"  Test samples: {args.num_test}")
    print(f"  Sequence length: {args.min_length}-{args.max_length}")

    # Generate and write each split
    for split, num_samples in [
        ("train", args.num_train),
        ("val", args.num_val),
        ("test", args.num_test),
    ]:
        print(f"\nGenerating {split} set...")
        samples = generate_samples(
            num_samples=num_samples,
            max_depth=args.max_depth,
            max_args=args.max_args,
            min_length=args.min_length,
            max_length=args.max_length,
        )

        filepath = output_dir / f"basic_{split}.tsv"
        write_to_file(samples, filepath)

    print(f"\n✓ Dataset generation complete!")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
