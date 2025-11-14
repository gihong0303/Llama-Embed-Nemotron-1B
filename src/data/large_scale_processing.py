"""
Large-Scale Dataset Processing Pipeline

Handles 16.1M+ scale datasets efficiently:
- Streaming JSONL processing (constant memory)
- Distributed hard negative mining
- Chunked synthetic data generation
- Dataset merging and deduplication

Based on paper: 16.1M samples (7.7M non-synthetic + 8.4M synthetic)
"""

import json
import os
from typing import List, Dict, Iterator, Optional, Tuple
from pathlib import Path
import hashlib
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


class StreamingJSONLReader:
    """
    Memory-efficient streaming reader for large JSONL files.

    Yields one sample at a time without loading entire file into memory.
    """

    def __init__(self, file_path: str, buffer_size: int = 8192):
        self.file_path = file_path
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[Dict]:
        """Iterate over samples in JSONL file."""
        with open(self.file_path, 'r', encoding='utf-8', buffering=self.buffer_size) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def count(self) -> int:
        """Count total samples (requires full file scan)."""
        count = 0
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count


class LargeScaleDatasetProcessor:
    """
    Process datasets at 16.1M+ scale efficiently.

    Features:
    - Streaming processing (constant memory)
    - Parallel processing
    - Progress tracking
    - Deduplication
    """

    def __init__(
        self,
        output_dir: str,
        chunk_size: int = 100000,  # Process 100k at a time
        num_workers: int = 8,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.num_workers = num_workers

    def merge_datasets(
        self,
        input_files: List[str],
        output_file: str,
        deduplicate: bool = True,
        show_progress: bool = True,
    ) -> Dict[str, int]:
        """
        Merge multiple JSONL files into one.

        Args:
            input_files: List of input JSONL paths
            output_file: Output JSONL path
            deduplicate: Remove duplicates based on query+positive hash
            show_progress: Show progress bar

        Returns:
            stats: {"total": N, "duplicates": M, "unique": K}
        """
        seen_hashes = set() if deduplicate else None
        total = 0
        duplicates = 0
        unique = 0

        print(f"Merging {len(input_files)} files into {output_file}...")

        # Count total for progress bar
        if show_progress:
            total_samples = sum(
                StreamingJSONLReader(f).count() for f in input_files
            )
            pbar = tqdm(total=total_samples, desc="Merging")

        with open(output_file, 'w', encoding='utf-8') as out_f:
            for input_file in input_files:
                reader = StreamingJSONLReader(input_file)

                for sample in reader:
                    total += 1

                    if deduplicate:
                        # Hash based on query + positive (core content)
                        content = sample['query'] + sample['positive']
                        content_hash = hashlib.md5(content.encode()).hexdigest()

                        if content_hash in seen_hashes:
                            duplicates += 1
                            if show_progress:
                                pbar.update(1)
                            continue

                        seen_hashes.add(content_hash)

                    unique += 1
                    out_f.write(json.dumps(sample, ensure_ascii=False) + '\n')

                    if show_progress:
                        pbar.update(1)

        if show_progress:
            pbar.close()

        stats = {
            "total": total,
            "duplicates": duplicates,
            "unique": unique,
        }

        print(f"Merge complete: {unique:,} unique samples (removed {duplicates:,} duplicates)")
        return stats

    def split_dataset(
        self,
        input_file: str,
        train_ratio: float = 0.9,
        val_ratio: float = 0.1,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Tuple[str, str]:
        """
        Split dataset into train/val.

        Args:
            input_file: Input JSONL file
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            shuffle: Shuffle before split
            seed: Random seed

        Returns:
            (train_file, val_file): Paths to output files
        """
        import random
        random.seed(seed)

        # Read all samples (if dataset fits in memory)
        # For very large datasets, use reservoir sampling
        reader = StreamingJSONLReader(input_file)
        samples = list(reader)

        print(f"Loaded {len(samples):,} samples")

        if shuffle:
            random.shuffle(samples)

        # Split
        train_size = int(len(samples) * train_ratio)

        train_samples = samples[:train_size]
        val_samples = samples[train_size:]

        # Write splits
        base_name = Path(input_file).stem
        train_file = str(self.output_dir / f"{base_name}_train.jsonl")
        val_file = str(self.output_dir / f"{base_name}_val.jsonl")

        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in train_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        with open(val_file, 'w', encoding='utf-8') as f:
            for sample in val_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"Split complete:")
        print(f"  Train: {len(train_samples):,} samples → {train_file}")
        print(f"  Val:   {len(val_samples):,} samples → {val_file}")

        return train_file, val_file

    def process_in_chunks(
        self,
        input_file: str,
        process_fn,
        output_file: str,
        show_progress: bool = True,
    ):
        """
        Process large file in chunks.

        Args:
            input_file: Input JSONL
            process_fn: Function to apply to each sample
            output_file: Output JSONL
            show_progress: Show progress
        """
        reader = StreamingJSONLReader(input_file)

        if show_progress:
            total = reader.count()
            pbar = tqdm(total=total, desc="Processing")

        with open(output_file, 'w', encoding='utf-8') as out_f:
            chunk = []

            for sample in StreamingJSONLReader(input_file):
                chunk.append(sample)

                if len(chunk) >= self.chunk_size:
                    # Process chunk
                    processed = [process_fn(s) for s in chunk]

                    # Write
                    for s in processed:
                        if s is not None:
                            out_f.write(json.dumps(s, ensure_ascii=False) + '\n')

                    if show_progress:
                        pbar.update(len(chunk))

                    chunk = []

            # Process remaining
            if chunk:
                processed = [process_fn(s) for s in chunk]
                for s in processed:
                    if s is not None:
                        out_f.write(json.dumps(s, ensure_ascii=False) + '\n')

                if show_progress:
                    pbar.update(len(chunk))

        if show_progress:
            pbar.close()

    def distributed_hard_negative_mining(
        self,
        input_file: str,
        corpus_file: str,
        output_file: str,
        mining_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        k: int = 4,
    ):
        """
        Perform hard negative mining on large dataset using distributed processing.

        For 16.1M samples, this processes in chunks to avoid OOM.

        Args:
            input_file: Input JSONL (queries and positives)
            corpus_file: Corpus file (one doc per line)
            output_file: Output with hard negatives
            mining_model: Model for mining
            k: Number of negatives per query
        """
        try:
            from .hard_negative_mining import HardNegativeMiner
        except ImportError:
            from src.data.hard_negative_mining import HardNegativeMiner

        print(f"Distributed hard negative mining on {input_file}...")
        print(f"  Mining model: {mining_model}")
        print(f"  Negatives per sample: {k}")

        # Load corpus (this might be large, consider chunking for very large corpora)
        print("Loading corpus...")
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus = [line.strip() for line in f if line.strip()]
        print(f"  Corpus size: {len(corpus):,} documents")

        # Initialize miner
        miner = HardNegativeMiner(model_name=mining_model)

        # Process in chunks
        reader = StreamingJSONLReader(input_file)
        total = reader.count()

        with open(output_file, 'w', encoding='utf-8') as out_f:
            chunk = []

            for sample in tqdm(StreamingJSONLReader(input_file), total=total, desc="Mining"):
                chunk.append(sample)

                if len(chunk) >= self.chunk_size:
                            # Mine for this chunk
                    queries = [s['query'] for s in chunk]
                    positives = [s['positive'] for s in chunk]

                    negatives = miner.mine(
                        queries=queries,
                        positives=positives,
                        corpus=corpus,
                        k=k,
                        show_progress=False,
                    )

                    # Write chunk with negatives
                    for sample, neg_list in zip(chunk, negatives):
                        sample['negatives'] = neg_list
                        out_f.write(json.dumps(sample, ensure_ascii=False) + '\n')

                    chunk = []

            # Process remaining
            if chunk:
                queries = [s['query'] for s in chunk]
                positives = [s['positive'] for s in chunk]

                negatives = miner.mine(
                    queries=queries,
                    positives=positives,
                    corpus=corpus,
                    k=k,
                    show_progress=False,
                )

                for sample, neg_list in zip(chunk, negatives):
                    sample['negatives'] = neg_list
                    out_f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"Mining complete: {output_file}")

    def create_paper_scale_dataset(
        self,
        retrieval_files: List[str],
        sts_files: List[str],
        classification_files: List[str],
        output_stage1: str,
        output_stage2: str,
        stage1_ratio: float = 0.7,
    ):
        """
        Create paper-scale dataset (16.1M total).

        Stage 1 (70%): Retrieval-focused
        Stage 2 (30%): Multi-task (retrieval + STS + classification)

        Args:
            retrieval_files: List of retrieval JSONL files
            sts_files: List of STS JSONL files
            classification_files: List of classification JSONL files
            output_stage1: Output for stage 1
            output_stage2: Output for stage 2
            stage1_ratio: Fraction of retrieval data for stage 1
        """
        print("Creating paper-scale dataset (16.1M+)...")

        # Merge retrieval data
        retrieval_merged = str(self.output_dir / "retrieval_merged.jsonl")
        self.merge_datasets(retrieval_files, retrieval_merged, deduplicate=True)

        retrieval_count = StreamingJSONLReader(retrieval_merged).count()

        # Split retrieval data: 70% for stage 1, 30% for stage 2
        stage1_size = int(retrieval_count * stage1_ratio)

        print(f"\nSplitting retrieval data:")
        print(f"  Total: {retrieval_count:,}")
        print(f"  Stage 1: {stage1_size:,} (70%)")
        print(f"  Stage 2: {retrieval_count - stage1_size:,} (30%)")

        # Read and split
        retrieval_samples = list(StreamingJSONLReader(retrieval_merged))

        stage1_retrieval = retrieval_samples[:stage1_size]
        stage2_retrieval = retrieval_samples[stage1_size:]

        # Write stage 1 (pure retrieval)
        with open(output_stage1, 'w', encoding='utf-8') as f:
            for sample in stage1_retrieval:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        # Merge stage 2 (retrieval + STS + classification)
        stage2_temp = str(self.output_dir / "stage2_temp.jsonl")

        # Add remaining retrieval
        with open(stage2_temp, 'w', encoding='utf-8') as f:
            for sample in stage2_retrieval:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        # Add STS and classification
        all_stage2_files = [stage2_temp] + sts_files + classification_files
        self.merge_datasets(all_stage2_files, output_stage2, deduplicate=True)

        # Clean up
        os.remove(stage2_temp)
        os.remove(retrieval_merged)

        # Final stats
        stage1_count = StreamingJSONLReader(output_stage1).count()
        stage2_count = StreamingJSONLReader(output_stage2).count()
        total = stage1_count + stage2_count

        print(f"\nDataset creation complete:")
        print(f"  Stage 1: {stage1_count:,} samples ({stage1_count/total*100:.1f}%)")
        print(f"  Stage 2: {stage2_count:,} samples ({stage2_count/total*100:.1f}%)")
        print(f"  Total: {total:,} samples")


# Example usage
if __name__ == "__main__":
    # Create processor
    processor = LargeScaleDatasetProcessor(
        output_dir="data/processed",
        chunk_size=100000,
    )

    # Example: Merge multiple retrieval datasets
    retrieval_files = [
        "data/raw/msmarco_retrieval.jsonl",
        "data/raw/nq_retrieval.jsonl",
        "data/raw/squad_retrieval.jsonl",
        "data/synthetic/generated_retrieval.jsonl",
    ]

    # Merge and deduplicate
    merged = "data/processed/retrieval_merged.jsonl"
    processor.merge_datasets(retrieval_files, merged, deduplicate=True)

    # Add hard negatives
    processor.distributed_hard_negative_mining(
        input_file=merged,
        corpus_file="data/corpus.txt",
        output_file="data/processed/retrieval_with_negatives.jsonl",
        k=4,
    )

    # Split for training
    train_file, val_file = processor.split_dataset(
        "data/processed/retrieval_with_negatives.jsonl",
        train_ratio=0.95,
        val_ratio=0.05,
    )

    print(f"\nReady for training:")
    print(f"  Train: {train_file}")
    print(f"  Val: {val_file}")
