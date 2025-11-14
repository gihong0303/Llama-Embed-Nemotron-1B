"""
Basic Usage Examples for Llama-Embed-Nemotron-1B

This script demonstrates how to use the trained embedding model for various tasks.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer
from src.models.embedding_model import InstructionAwareEmbeddingModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def example_retrieval():
    """Example: Retrieval task"""
    print("\n" + "="*80)
    print("EXAMPLE 1: RETRIEVAL")
    print("="*80)

    # Load model
    model_path = "./outputs/stage2/best_model"  # Change to your model path
    model = InstructionAwareEmbeddingModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    # Query and documents
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital and largest city of France.",
        "Berlin is the capital of Germany.",
        "The Eiffel Tower is located in Paris.",
        "France is a country in Western Europe.",
        "London is the capital of the United Kingdom.",
    ]

    # Task instruction
    instruction = "Retrieve relevant passages for this query"

    # Encode query
    query_embed = model.encode(
        query,
        instruction=instruction,
        tokenizer=tokenizer,
        normalize=True,
        convert_to_numpy=True,
    )

    # Encode documents (no instruction for documents)
    doc_embeds = model.encode(
        documents,
        instruction="",
        tokenizer=tokenizer,
        normalize=True,
        convert_to_numpy=True,
    )

    # Compute similarities
    similarities = cosine_similarity(query_embed.reshape(1, -1), doc_embeds)[0]

    # Rank documents
    ranked_indices = np.argsort(-similarities)

    print(f"\nQuery: {query}\n")
    print("Ranked Documents:")
    for i, idx in enumerate(ranked_indices):
        print(f"{i+1}. (score: {similarities[idx]:.4f}) {documents[idx]}")


def example_semantic_similarity():
    """Example: Semantic Textual Similarity"""
    print("\n" + "="*80)
    print("EXAMPLE 2: SEMANTIC TEXTUAL SIMILARITY")
    print("="*80)

    # Load model
    model_path = "./outputs/stage2/best_model"
    model = InstructionAwareEmbeddingModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    # Text pairs
    text_pairs = [
        ("The cat sat on the mat.", "A feline rested on the rug."),
        ("I love machine learning.", "Artificial intelligence is fascinating."),
        ("The weather is nice today.", "I bought a new car yesterday."),
    ]

    # Task instruction
    instruction = "Retrieve semantically similar text"

    print("\nComputing semantic similarity:\n")

    for text1, text2 in text_pairs:
        # Encode both texts
        embeds = model.encode(
            [text1, text2],
            instruction=instruction,
            tokenizer=tokenizer,
            normalize=True,
            convert_to_numpy=True,
        )

        # Compute cosine similarity
        similarity = cosine_similarity(embeds[0:1], embeds[1:2])[0][0]

        print(f"Text 1: {text1}")
        print(f"Text 2: {text2}")
        print(f"Similarity: {similarity:.4f}\n")


def example_classification():
    """Example: Text Classification via Embedding Similarity"""
    print("\n" + "="*80)
    print("EXAMPLE 3: TEXT CLASSIFICATION")
    print("="*80)

    # Load model
    model_path = "./outputs/stage2/best_model"
    model = InstructionAwareEmbeddingModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    # Texts to classify
    texts = [
        "The stock market reached a new high today.",
        "Scientists discovered a new species of frog.",
        "The Lakers won the championship game.",
        "New research shows promising results for cancer treatment.",
    ]

    # Class labels
    labels = ["Business", "Science", "Sports"]

    # Task instruction
    instruction = "Classify the topic of this text"

    # Encode texts
    text_embeds = model.encode(
        texts,
        instruction=instruction,
        tokenizer=tokenizer,
        normalize=True,
        convert_to_numpy=True,
    )

    # Encode labels
    label_embeds = model.encode(
        labels,
        instruction=instruction,
        tokenizer=tokenizer,
        normalize=True,
        convert_to_numpy=True,
    )

    # Classify by finding nearest label
    similarities = cosine_similarity(text_embeds, label_embeds)
    predictions = np.argmax(similarities, axis=1)

    print("\nClassification Results:\n")
    for text, pred_idx, sims in zip(texts, predictions, similarities):
        pred_label = labels[pred_idx]
        confidence = sims[pred_idx]

        print(f"Text: {text}")
        print(f"Predicted: {pred_label} (confidence: {confidence:.4f})")
        print(f"All scores: {', '.join([f'{l}: {s:.3f}' for l, s in zip(labels, sims)])}\n")


def example_clustering():
    """Example: Document Clustering"""
    print("\n" + "="*80)
    print("EXAMPLE 4: DOCUMENT CLUSTERING")
    print("="*80)

    # Load model
    model_path = "./outputs/stage2/best_model"
    model = InstructionAwareEmbeddingModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    # Documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "The stock market is volatile today.",
        "Neural networks are inspired by the human brain.",
        "Interest rates are expected to rise.",
        "Deep learning has revolutionized computer vision.",
        "The Federal Reserve announced new monetary policy.",
    ]

    # Task instruction
    instruction = "Identify the topic or theme of the text"

    # Encode documents
    doc_embeds = model.encode(
        documents,
        instruction=instruction,
        tokenizer=tokenizer,
        normalize=True,
        convert_to_numpy=True,
    )

    # Simple clustering using K-means
    from sklearn.cluster import KMeans

    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(doc_embeds)

    print(f"\nClustered into {n_clusters} groups:\n")
    for cluster_id in range(n_clusters):
        print(f"Cluster {cluster_id + 1}:")
        cluster_docs = [doc for doc, label in zip(documents, cluster_labels) if label == cluster_id]
        for doc in cluster_docs:
            print(f"  - {doc}")
        print()


def example_batch_encoding():
    """Example: Efficient Batch Encoding"""
    print("\n" + "="*80)
    print("EXAMPLE 5: BATCH ENCODING")
    print("="*80)

    # Load model
    model_path = "./outputs/stage2/best_model"
    model = InstructionAwareEmbeddingModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    # Large batch of texts
    texts = [f"This is document number {i}." for i in range(100)]

    instruction = "Retrieve relevant passages for this query"

    print(f"Encoding {len(texts)} documents in batches...\n")

    # Encode with batch processing
    embeddings = model.encode(
        texts,
        instruction=instruction,
        tokenizer=tokenizer,
        batch_size=32,
        normalize=True,
        convert_to_numpy=True,
        show_progress=True,
    )

    print(f"\nEncoded {len(texts)} documents")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")


if __name__ == "__main__":
    # Run all examples (comment out if you don't have a trained model yet)
    # example_retrieval()
    # example_semantic_similarity()
    # example_classification()
    # example_clustering()
    # example_batch_encoding()

    print("\n" + "="*80)
    print("To run examples, uncomment the function calls above and provide a trained model path.")
    print("="*80 + "\n")
