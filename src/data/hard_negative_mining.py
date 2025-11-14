"""
Hard Negative Mining for Contrastive Learning

Based on NV-Retriever strategy (referenced in Llama-Embed-Nemotron paper):
- Use embedding model to find top-k similar documents
- Filter out negatives that are too similar to positive (>= 95% of positive similarity)
- This avoids false negatives (documents that are actually relevant)

Usage:
    >>> miner = HardNegativeMiner(model_name="sentence-transformers/all-MiniLM-L6-v2")
    >>> queries = ["What is the capital of France?"]
    >>> positives = ["Paris is the capital of France."]
    >>> corpus = ["Berlin is the capital of Germany.", "Rome is in Italy.", ...]
    >>> negatives = miner.mine(queries, positives, corpus, k=4)
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Union
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm


class HardNegativeMiner:
    """
    Hard negative mining using dense retrieval.

    Strategy (from NV-Retriever):
    1. Encode all corpus documents
    2. For each query, find top-k most similar documents
    3. Filter out documents where similarity(query, doc) >= 0.95 * similarity(query, positive)
       (these might be false negatives)
    4. Return filtered hard negatives

    Args:
        model_name: Embedding model for mining (e.g., "e5-mistral-7b-instruct")
        device: Device to run on
        batch_size: Batch size for encoding
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 128,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.batch_size = batch_size

        print(f"Loading mining model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.model.eval()

        print(f"Mining model loaded on {device}")

    def encode(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: List of texts to encode
            show_progress: Show progress bar

        Returns:
            embeddings: [num_texts, embedding_dim]
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity
        )
        return embeddings

    def mine(
        self,
        queries: List[str],
        positives: List[str],
        corpus: List[str],
        k: int = 10,
        similarity_threshold: float = 0.95,
        show_progress: bool = True,
    ) -> List[List[str]]:
        """
        Mine hard negatives for each query.

        Args:
            queries: List of queries
            positives: List of positive documents (one per query)
            corpus: Corpus of documents to mine from
            k: Number of hard negatives to return per query
            similarity_threshold: Filter threshold (default: 0.95)
                                 Negatives with sim >= threshold * pos_sim are filtered
            show_progress: Show progress bar

        Returns:
            hard_negatives: List of lists, where hard_negatives[i] contains
                           hard negatives for queries[i]
        """
        assert len(queries) == len(positives), "Queries and positives must have same length"

        print(f"Mining hard negatives from corpus of {len(corpus)} documents...")

        # Encode queries and positives
        print("Encoding queries...")
        query_embeds = self.encode(queries, show_progress=show_progress)

        print("Encoding positives...")
        pos_embeds = self.encode(positives, show_progress=show_progress)

        print("Encoding corpus...")
        corpus_embeds = self.encode(corpus, show_progress=show_progress)

        # Compute positive similarities
        pos_sims = np.sum(query_embeds * pos_embeds, axis=1)  # [num_queries]

        # Build FAISS index for fast retrieval
        print("Building FAISS index...")
        embedding_dim = corpus_embeds.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine sim with normalized vectors)
        index.add(corpus_embeds.astype(np.float32))

        # Search for top-k candidates (retrieve more than k to account for filtering)
        search_k = min(k * 5, len(corpus))  # Retrieve 5x more candidates
        print(f"Searching for top-{search_k} candidates per query...")
        scores, indices = index.search(query_embeds.astype(np.float32), search_k)

        # Filter and select hard negatives
        hard_negatives = []
        num_filtered = 0
        num_total_candidates = 0

        for i, (query_indices, query_scores, pos_sim) in enumerate(zip(indices, scores, pos_sims)):
            # Filter threshold: keep negatives with sim < threshold * pos_sim
            threshold = similarity_threshold * pos_sim

            filtered_negatives = []
            for idx, score in zip(query_indices, query_scores):
                num_total_candidates += 1

                # Skip if this is the positive document (exact match)
                if corpus[idx] == positives[i]:
                    num_filtered += 1
                    continue

                # Skip if too similar to positive (potential false negative)
                if score >= threshold:
                    num_filtered += 1
                    continue

                filtered_negatives.append(corpus[idx])

                # Stop once we have k negatives
                if len(filtered_negatives) >= k:
                    break

            # If not enough negatives after filtering, pad with lower-ranked candidates
            if len(filtered_negatives) < k:
                for idx in query_indices:
                    if corpus[idx] not in filtered_negatives and corpus[idx] != positives[i]:
                        filtered_negatives.append(corpus[idx])
                        if len(filtered_negatives) >= k:
                            break

            hard_negatives.append(filtered_negatives[:k])

        filter_rate = num_filtered / num_total_candidates if num_total_candidates > 0 else 0
        print(f"Mining complete. Filtered {num_filtered}/{num_total_candidates} candidates ({filter_rate:.1%})")

        return hard_negatives

    def mine_batched(
        self,
        dataset: List[Dict[str, str]],
        corpus: List[str],
        k: int = 10,
        query_key: str = "query",
        positive_key: str = "positive",
        show_progress: bool = True,
    ) -> List[Dict[str, Union[str, List[str]]]]:
        """
        Mine hard negatives for a dataset of query-positive pairs.

        Args:
            dataset: List of dicts with "query" and "positive" keys
            corpus: Corpus to mine from
            k: Number of negatives per sample
            query_key: Key for query in dataset dicts
            positive_key: Key for positive in dataset dicts
            show_progress: Show progress bar

        Returns:
            dataset_with_negatives: Dataset with added "negatives" key
        """
        queries = [sample[query_key] for sample in dataset]
        positives = [sample[positive_key] for sample in dataset]

        # Mine negatives
        negatives = self.mine(
            queries,
            positives,
            corpus,
            k=k,
            show_progress=show_progress,
        )

        # Add negatives to dataset
        dataset_with_negatives = []
        for sample, neg_list in zip(dataset, negatives):
            new_sample = sample.copy()
            new_sample["negatives"] = neg_list
            dataset_with_negatives.append(new_sample)

        return dataset_with_negatives


class RandomNegativeSampler:
    """
    Simple random negative sampler (baseline).

    For comparison with hard negative mining.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def sample(
        self,
        queries: List[str],
        positives: List[str],
        corpus: List[str],
        k: int = 10,
    ) -> List[List[str]]:
        """
        Sample random negatives for each query.

        Args:
            queries: List of queries
            positives: List of positive documents
            corpus: Corpus to sample from
            k: Number of negatives per query

        Returns:
            negatives: List of lists of negatives
        """
        negatives = []
        corpus_set = set(corpus)

        for pos in positives:
            # Filter out the positive
            available = list(corpus_set - {pos})

            # Sample k negatives
            sampled = self.rng.choice(available, size=min(k, len(available)), replace=False)
            negatives.append(sampled.tolist())

        return negatives


def create_negative_miner(
    mining_strategy: str = "hard",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    **kwargs
) -> Union[HardNegativeMiner, RandomNegativeSampler]:
    """
    Factory function to create a negative miner.

    Args:
        mining_strategy: One of ["hard", "random"]
        model_name: Model for hard negative mining
        **kwargs: Additional arguments

    Returns:
        Negative miner
    """
    if mining_strategy == "hard":
        return HardNegativeMiner(model_name=model_name, **kwargs)
    elif mining_strategy == "random":
        return RandomNegativeSampler(**kwargs)
    else:
        raise ValueError(f"Unknown mining strategy: {mining_strategy}")


# Example usage
if __name__ == "__main__":
    # Example
    miner = HardNegativeMiner(model_name="sentence-transformers/all-MiniLM-L6-v2")

    queries = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
    ]

    positives = [
        "Paris is the capital and largest city of France.",
        "Romeo and Juliet is a tragedy written by William Shakespeare.",
    ]

    corpus = [
        "Paris is the capital and largest city of France.",
        "Romeo and Juliet is a tragedy written by William Shakespeare.",
        "Berlin is the capital of Germany.",
        "London is the capital of the United Kingdom.",
        "Shakespeare was an English playwright and poet.",
        "France is a country in Western Europe.",
        "The Eiffel Tower is located in Paris.",
        "Hamlet is another play by Shakespeare.",
        "Rome is the capital of Italy.",
        "Tokyo is the capital of Japan.",
    ]

    hard_negatives = miner.mine(queries, positives, corpus, k=3)

    for i, (query, neg_list) in enumerate(zip(queries, hard_negatives)):
        print(f"\nQuery {i+1}: {query}")
        print(f"Hard negatives:")
        for j, neg in enumerate(neg_list):
            print(f"  {j+1}. {neg}")
