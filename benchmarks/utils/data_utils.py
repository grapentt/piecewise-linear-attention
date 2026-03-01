"""Data utilities for translation benchmarks."""

from typing import Dict, List, Tuple, Sequence

import torch
from torch.utils.data import Dataset
from datasets import load_dataset


def tokenize(sentence: str) -> List[str]:
    """Lowercase and split a raw sentence into whitespace tokens."""
    return sentence.strip().lower().split()


def make_dictionary(data, unk_threshold: int = 0) -> Dict[str, int]:
    """Create dictionary from tokenized sentences.

    Args:
        data: Iterable of token lists or raw strings
        unk_threshold: Words below this count threshold are excluded and replaced with UNK

    Returns:
        Dictionary mapping words to indices
    """
    # Count word frequencies
    word_frequencies = {}
    for sent in data:
        tokens = tokenize(sent) if isinstance(sent, str) else [w.lower() for w in sent]
        for word in tokens:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

    # Assign indices (special tokens first)
    word_to_ix = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
    for word, freq in word_frequencies.items():
        if freq > unk_threshold:
            word_to_ix[word] = len(word_to_ix)

    print(f"Dictionary contains {len(word_to_ix)} words (unk_threshold={unk_threshold})")
    return word_to_ix


def create_indices(sentences, word_to_ix, device):
    """Convert sentences to padded indices."""
    normalized = []
    for sentence in sentences:
        if isinstance(sentence, str):
            normalized.append(tokenize(sentence))
        else:
            normalized.append([w.lower() for w in sentence])

    lengths = [len(sentence) for sentence in normalized]
    longest_sequence = max(lengths)

    indices_batch = []
    for sentence in normalized:
        indices = [word_to_ix["<sos>"]]

        for word in sentence:
            if word in word_to_ix:
                indices.append(word_to_ix[word])
            else:
                indices.append(word_to_ix.get("<unk>", 0))

        indices.append(word_to_ix["<eos>"])

        # Pad to longest in batch
        for _ in range(longest_sequence - len(sentence)):
            indices.append(word_to_ix["<pad>"])

        indices_batch.append(indices)

    return torch.tensor(indices_batch).to(device)


class TranslationDataset(Dataset):
    """Translation dataset from HuggingFace."""

    def __init__(self, pairs: Sequence[Tuple[List[str], List[str]]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[List[str], List[str]]:
        return self.pairs[idx]


def prepare_dataset(
    dataset_name: str,
    subset_name: str,
    src_lang: str,
    tgt_lang: str,
    max_train_samples: int,
    max_eval_samples: int,
    max_seq_len: int = 128,
) -> Tuple[TranslationDataset, TranslationDataset, Dict[str, int], Dict[str, int]]:
    """Load and prepare HuggingFace translation dataset.

    Args:
        dataset_name: HuggingFace dataset name
        subset_name: Subset name
        src_lang: Source language code
        tgt_lang: Target language code
        max_train_samples: Max training samples
        max_eval_samples: Max eval samples
        max_seq_len: Max sequence length

    Returns:
        Tuple of (train_dataset, eval_dataset, src_dict, tgt_dict)
    """
    print(f"Loading dataset {dataset_name}/{subset_name}...")
    dataset = load_dataset(dataset_name, subset_name)

    max_tokens = max_seq_len - 2  # Reserve spots for <sos> and <eos>

    def extract_pairs(split_name: str, limit: int) -> List[Tuple[List[str], List[str]]]:
        if split_name not in dataset:
            return []

        split = dataset[split_name]
        if limit is not None:
            split = split.select(range(min(limit, len(split))))

        pairs = []
        for example in split:
            if "translation" in example:
                example = example["translation"]

            src = tokenize(example[src_lang])[:max_tokens]
            tgt = tokenize(example[tgt_lang])[:max_tokens]
            pairs.append((src, tgt))

        return pairs

    train_pairs = extract_pairs("train", max_train_samples)
    eval_split = "validation" if "validation" in dataset else "test"
    eval_pairs = extract_pairs(eval_split, max_eval_samples)

    print(f"✓ Loaded {len(train_pairs)} train pairs, {len(eval_pairs)} eval pairs")

    # Build vocabularies
    src_sentences = [pair[0] for pair in train_pairs]
    tgt_sentences = [pair[1] for pair in train_pairs]

    print("Building source vocabulary...")
    src_dict = make_dictionary(src_sentences)
    print("Building target vocabulary...")
    tgt_dict = make_dictionary(tgt_sentences)

    return (
        TranslationDataset(train_pairs),
        TranslationDataset(eval_pairs),
        src_dict,
        tgt_dict,
    )


def collate_fn(batch, src_dict, tgt_dict, device):
    """Collate function for DataLoader."""
    src_sentences, tgt_sentences = zip(*batch)
    src_indices = create_indices(src_sentences, src_dict, device)
    tgt_indices = create_indices(tgt_sentences, tgt_dict, device)
    return src_indices, tgt_indices
