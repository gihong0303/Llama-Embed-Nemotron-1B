"""
Dataset classes for text embedding training

Supports multiple task types:
- Retrieval: (query, positive_doc, negative_docs)
- STS: (text_a, text_b, similarity_score)
- Classification: (text, label, negative_labels)
- Bitext: (source_text, target_text, negative_targets)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import List, Dict, Optional, Union, Tuple
from transformers import PreTrainedTokenizer
import json
import random


class EmbeddingDataset(Dataset):
    """
    Generic dataset for embedding tasks.

    Data format:
        {
            "query": str,              # Query or text_a
            "positive": str,           # Positive document or text_b
            "negatives": List[str],    # List of negative documents
            "instruction": str,        # Task instruction (optional)
            "task_type": str,          # One of ["retrieval", "sts", "classification", "bitext"]
        }

    Args:
        data: List of data samples
        tokenizer: Tokenizer for encoding texts
        max_length: Maximum sequence length
        num_negatives: Number of negatives to use per sample
        instruction: Default instruction (can be overridden per sample)
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        num_negatives: int = 1,
        instruction: Optional[str] = None,
        task_type: str = "retrieval",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.default_instruction = instruction
        self.task_type = task_type

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.

        Returns:
            {
                "query_input_ids": [seq_len],
                "query_attention_mask": [seq_len],
                "pos_input_ids": [seq_len],
                "pos_attention_mask": [seq_len],
                "neg_input_ids": [num_negatives, seq_len],
                "neg_attention_mask": [num_negatives, seq_len],
                "task_type": str,
            }
        """
        sample = self.data[idx]

        query = sample["query"]
        positive = sample["positive"]
        negatives = sample.get("negatives", [])
        instruction = sample.get("instruction", self.default_instruction)
        task_type = sample.get("task_type", self.task_type)

        # Format query with instruction
        if instruction:
            query_formatted = f"Instruct: {instruction}\nQuery: {query}"
        else:
            query_formatted = query

        # For retrieval, positive doesn't get instruction
        # For STS and classification, positive also gets instruction
        if task_type in ["sts", "classification"]:
            if instruction:
                positive_formatted = f"Instruct: {instruction}\nQuery: {positive}"
            else:
                positive_formatted = positive
        else:
            positive_formatted = positive

        # Tokenize query
        query_encoded = self.tokenizer(
            query_formatted,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize positive
        pos_encoded = self.tokenizer(
            positive_formatted,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Sample and tokenize negatives
        if negatives:
            # Sample num_negatives from available negatives
            if len(negatives) >= self.num_negatives:
                sampled_negatives = random.sample(negatives, self.num_negatives)
            else:
                # If not enough negatives, repeat some
                sampled_negatives = negatives * (self.num_negatives // len(negatives) + 1)
                sampled_negatives = sampled_negatives[:self.num_negatives]

            # For retrieval, negatives don't get instruction
            # For others, they might
            if task_type in ["sts", "classification"]:
                if instruction:
                    negatives_formatted = [f"Instruct: {instruction}\nQuery: {neg}" for neg in sampled_negatives]
                else:
                    negatives_formatted = sampled_negatives
            else:
                negatives_formatted = sampled_negatives

            # Tokenize each negative
            neg_encoded = self.tokenizer(
                negatives_formatted,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            neg_input_ids = neg_encoded["input_ids"]  # [num_negatives, seq_len]
            neg_attention_mask = neg_encoded["attention_mask"]
        else:
            # No negatives provided
            neg_input_ids = torch.zeros((self.num_negatives, self.max_length), dtype=torch.long)
            neg_attention_mask = torch.zeros((self.num_negatives, self.max_length), dtype=torch.long)

        return {
            "query_input_ids": query_encoded["input_ids"].squeeze(0),
            "query_attention_mask": query_encoded["attention_mask"].squeeze(0),
            "pos_input_ids": pos_encoded["input_ids"].squeeze(0),
            "pos_attention_mask": pos_encoded["attention_mask"].squeeze(0),
            "neg_input_ids": neg_input_ids,
            "neg_attention_mask": neg_attention_mask,
            "task_type": task_type,
        }


class MultiTaskDataset(Dataset):
    """
    Dataset that combines multiple task-specific datasets.

    Args:
        datasets: List of (dataset, weight) tuples
        sampling_strategy: One of ["proportional", "uniform"]
    """

    def __init__(
        self,
        datasets: List[Tuple[EmbeddingDataset, float]],
        sampling_strategy: str = "proportional",
    ):
        self.datasets = [ds for ds, _ in datasets]
        self.weights = [w for _, w in datasets]
        self.sampling_strategy = sampling_strategy

        # Compute total length and cumulative weights
        if sampling_strategy == "proportional":
            # Length is weighted sum of dataset lengths
            self.length = sum(len(ds) * w for ds, w in zip(self.datasets, self.weights))
            self.length = int(self.length)
        else:
            # Uniform sampling: use max dataset length
            self.length = max(len(ds) for ds in self.datasets)

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

        print(f"MultiTaskDataset created with {len(self.datasets)} datasets")
        print(f"  Sampling strategy: {sampling_strategy}")
        print(f"  Total length: {self.length}")
        for i, (ds, w) in enumerate(zip(self.datasets, self.weights)):
            print(f"  Dataset {i+1}: {len(ds)} samples, weight={w:.3f}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict:
        """
        Sample from one of the datasets based on weights.
        """
        # Sample dataset based on weights
        dataset_idx = random.choices(range(len(self.datasets)), weights=self.weights, k=1)[0]
        dataset = self.datasets[dataset_idx]

        # Sample random item from selected dataset
        item_idx = random.randint(0, len(dataset) - 1)

        return dataset[item_idx]


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save data to JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def create_retrieval_dataset(
    queries: List[str],
    positives: List[str],
    negatives: List[List[str]],
    instruction: str = "Retrieve relevant passages for this query",
) -> List[Dict]:
    """
    Create retrieval dataset from queries, positives, and negatives.

    Args:
        queries: List of queries
        positives: List of positive documents (one per query)
        negatives: List of negative lists (one list per query)
        instruction: Task instruction

    Returns:
        data: List of data samples in standard format
    """
    assert len(queries) == len(positives) == len(negatives)

    data = []
    for query, pos, negs in zip(queries, positives, negatives):
        data.append({
            "query": query,
            "positive": pos,
            "negatives": negs,
            "instruction": instruction,
            "task_type": "retrieval",
        })

    return data


def create_sts_dataset(
    texts_a: List[str],
    texts_b: List[str],
    negatives_a: List[List[str]],
    negatives_b: Optional[List[List[str]]] = None,
    instruction: str = "Retrieve semantically similar text",
) -> List[Dict]:
    """
    Create STS dataset from text pairs and negatives.

    Args:
        texts_a: List of first texts
        texts_b: List of second texts (semantically similar to texts_a)
        negatives_a: Negatives for texts_a
        negatives_b: Negatives for texts_b (if None, use negatives_a)
        instruction: Task instruction

    Returns:
        data: List of data samples
    """
    if negatives_b is None:
        negatives_b = negatives_a

    assert len(texts_a) == len(texts_b) == len(negatives_a) == len(negatives_b)

    data = []
    for text_a, text_b, negs_a, negs_b in zip(texts_a, texts_b, negatives_a, negatives_b):
        # Create bidirectional pairs
        data.append({
            "query": text_a,
            "positive": text_b,
            "negatives": negs_a,
            "instruction": instruction,
            "task_type": "sts",
        })
        data.append({
            "query": text_b,
            "positive": text_a,
            "negatives": negs_b,
            "instruction": instruction,
            "task_type": "sts",
        })

    return data


def create_classification_dataset(
    texts: List[str],
    labels: List[str],
    label_pool: List[str],
    num_negative_labels: int = 4,
    instruction: str = "Classify the topic of this text",
) -> List[Dict]:
    """
    Create classification dataset.

    Args:
        texts: List of texts
        labels: List of labels (one per text)
        label_pool: Pool of all possible labels
        num_negative_labels: Number of negative labels to sample
        instruction: Task instruction

    Returns:
        data: List of data samples
    """
    assert len(texts) == len(labels)

    data = []
    for text, label in zip(texts, labels):
        # Sample negative labels
        negative_labels = [l for l in label_pool if l != label]
        if len(negative_labels) >= num_negative_labels:
            sampled_negatives = random.sample(negative_labels, num_negative_labels)
        else:
            sampled_negatives = negative_labels

        data.append({
            "query": text,
            "positive": label,
            "negatives": sampled_negatives,
            "instruction": instruction,
            "task_type": "classification",
        })

    return data


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched tensors
    """
    # Stack all tensors
    query_input_ids = torch.stack([item["query_input_ids"] for item in batch])
    query_attention_mask = torch.stack([item["query_attention_mask"] for item in batch])
    pos_input_ids = torch.stack([item["pos_input_ids"] for item in batch])
    pos_attention_mask = torch.stack([item["pos_attention_mask"] for item in batch])
    neg_input_ids = torch.stack([item["neg_input_ids"] for item in batch])
    neg_attention_mask = torch.stack([item["neg_attention_mask"] for item in batch])

    task_types = [item["task_type"] for item in batch]

    return {
        "query_input_ids": query_input_ids,
        "query_attention_mask": query_attention_mask,
        "pos_input_ids": pos_input_ids,
        "pos_attention_mask": pos_attention_mask,
        "neg_input_ids": neg_input_ids,
        "neg_attention_mask": neg_attention_mask,
        "task_types": task_types,
    }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    distributed: bool = False,
) -> DataLoader:
    """
    Create DataLoader for training with optional DistributedSampler.

    Args:
        dataset: Dataset
        batch_size: Batch size per GPU
        shuffle: Whether to shuffle (ignored if distributed=True)
        num_workers: Number of worker processes
        distributed: Whether to use DistributedSampler for multi-GPU training

    Returns:
        DataLoader

    Note:
        When distributed=True, DistributedSampler will automatically shard
        the dataset across all GPUs. Each GPU will see a different subset.
        Call sampler.set_epoch(epoch) at the start of each epoch for proper shuffling.
    """
    # Use DistributedSampler in distributed mode
    if distributed:
        import torch.distributed as dist

        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,  # Shuffle within each epoch
            drop_last=False,  # Don't drop last incomplete batch
        )
        # Don't use shuffle argument when using a sampler
        shuffle = False
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,  # Keep all samples
    )
