"""
Evaluate Embedding Model

Evaluate the trained embedding model on various tasks.

Usage:
    # Basic evaluation
    python scripts/evaluate.py \
        --model_path outputs/stage2/best_model \
        --eval_data data/eval.jsonl

    # MTEB evaluation
    python scripts/evaluate.py \
        --model_path outputs/stage2/best_model \
        --use_mteb \
        --mteb_tasks "STSBenchmark,SICK"
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import numpy as np
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score
import json

from src.models.embedding_model import InstructionAwareEmbeddingModel
from src.data.dataset import load_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Embedding Model")

    # Model
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")

    # Data
    parser.add_argument("--eval_data", type=str, default=None,
                       help="Path to evaluation data (JSONL)")

    # MTEB
    parser.add_argument("--use_mteb", action="store_true",
                       help="Run MTEB evaluation")
    parser.add_argument("--mteb_tasks", type=str, default=None,
                       help="Comma-separated list of MTEB tasks (None = all)")

    # Settings
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for encoding")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")

    # Output
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory for results")

    return parser.parse_args()


def evaluate_retrieval(model, tokenizer, eval_data, batch_size=64):
    """
    Evaluate retrieval performance.

    Metrics:
    - Recall@k (k=1, 5, 10)
    - MRR (Mean Reciprocal Rank)
    """
    print("\nEvaluating retrieval task...")

    queries = [sample["query"] for sample in eval_data]
    positives = [sample["positive"] for sample in eval_data]
    all_negatives = [sample.get("negatives", []) for sample in eval_data]

    # Get instruction
    instruction = eval_data[0].get("instruction", "Retrieve relevant passages for this query")

    # Encode queries
    print("  Encoding queries...")
    query_embeds = model.encode(
        queries,
        instruction=instruction,
        tokenizer=tokenizer,
        batch_size=batch_size,
        normalize=True,
        convert_to_numpy=True,
    )

    # Encode positives and negatives
    print("  Encoding documents...")
    all_docs = []
    doc_to_idx = {}

    for i, (pos, negs) in enumerate(zip(positives, all_negatives)):
        # Positive
        if pos not in doc_to_idx:
            doc_to_idx[pos] = len(all_docs)
            all_docs.append(pos)

        # Negatives
        for neg in negs:
            if neg not in doc_to_idx:
                doc_to_idx[neg] = len(all_docs)
                all_docs.append(neg)

    doc_embeds = model.encode(
        all_docs,
        instruction="",  # No instruction for documents
        tokenizer=tokenizer,
        batch_size=batch_size,
        normalize=True,
        convert_to_numpy=True,
    )

    # Compute metrics
    recalls = {1: [], 5: [], 10: []}
    mrrs = []

    for i, (query, pos) in enumerate(zip(queries, positives)):
        query_embed = query_embeds[i:i+1]

        # Get candidate docs for this query
        candidates = [pos] + all_negatives[i]
        candidate_indices = [doc_to_idx[doc] for doc in candidates]
        candidate_embeds = doc_embeds[candidate_indices]

        # Compute similarities
        sims = cosine_similarity(query_embed, candidate_embeds)[0]

        # Rank
        ranked_indices = np.argsort(-sims)

        # Positive is at index 0
        pos_rank = np.where(ranked_indices == 0)[0][0] + 1

        # Recall@k
        for k in [1, 5, 10]:
            if pos_rank <= k:
                recalls[k].append(1.0)
            else:
                recalls[k].append(0.0)

        # MRR
        mrrs.append(1.0 / pos_rank)

    results = {
        "recall@1": np.mean(recalls[1]),
        "recall@5": np.mean(recalls[5]),
        "recall@10": np.mean(recalls[10]),
        "mrr": np.mean(mrrs),
    }

    return results


def evaluate_sts(model, tokenizer, eval_data, batch_size=64):
    """
    Evaluate semantic textual similarity.

    Metric: Spearman correlation between predicted and true similarity.
    """
    print("\nEvaluating STS task...")

    from scipy.stats import spearmanr

    texts_a = [sample["query"] for sample in eval_data]
    texts_b = [sample["positive"] for sample in eval_data]

    instruction = eval_data[0].get("instruction", "Retrieve semantically similar text")

    # Encode
    print("  Encoding texts...")
    embeds_a = model.encode(texts_a, instruction=instruction, tokenizer=tokenizer,
                           batch_size=batch_size, normalize=True, convert_to_numpy=True)
    embeds_b = model.encode(texts_b, instruction=instruction, tokenizer=tokenizer,
                           batch_size=batch_size, normalize=True, convert_to_numpy=True)

    # Compute similarities
    pred_sims = np.sum(embeds_a * embeds_b, axis=1)

    # If we have ground truth similarity scores
    if "similarity" in eval_data[0]:
        true_sims = [sample["similarity"] for sample in eval_data]
        corr, _ = spearmanr(pred_sims, true_sims)
        results = {"spearman_correlation": corr}
    else:
        # Otherwise, just compute mean similarity
        results = {"mean_similarity": np.mean(pred_sims)}

    return results


def evaluate_classification(model, tokenizer, eval_data, batch_size=64):
    """
    Evaluate classification by comparing text embedding to label embeddings.

    Metric: Accuracy
    """
    print("\nEvaluating classification task...")

    texts = [sample["query"] for sample in eval_data]
    true_labels = [sample["positive"] for sample in eval_data]

    # Get all unique labels
    all_labels = list(set(true_labels))
    print(f"  Number of classes: {len(all_labels)}")

    instruction = eval_data[0].get("instruction", "Classify the topic of this text")

    # Encode texts
    print("  Encoding texts...")
    text_embeds = model.encode(texts, instruction=instruction, tokenizer=tokenizer,
                              batch_size=batch_size, normalize=True, convert_to_numpy=True)

    # Encode labels
    print("  Encoding labels...")
    label_embeds = model.encode(all_labels, instruction=instruction, tokenizer=tokenizer,
                               batch_size=batch_size, normalize=True, convert_to_numpy=True)

    # Predict by finding nearest label
    sims = cosine_similarity(text_embeds, label_embeds)
    pred_indices = np.argmax(sims, axis=1)
    pred_labels = [all_labels[i] for i in pred_indices]

    # Compute accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average="weighted")

    results = {
        "accuracy": accuracy,
        "f1_weighted": f1,
    }

    return results


def run_mteb_evaluation(model_path, tasks=None):
    """
    Run MTEB (Massive Text Embedding Benchmark) evaluation.

    Requires: pip install mteb
    """
    try:
        from mteb import MTEB
    except ImportError:
        print("Error: MTEB not installed. Install with: pip install mteb")
        return None

    print("\nRunning MTEB evaluation...")

    # Wrap model for MTEB
    class MTEBModel:
        def __init__(self, model_path):
            self.model = InstructionAwareEmbeddingModel.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        def encode(self, sentences, **kwargs):
            # MTEB expects a simple encode interface
            return self.model.encode(
                sentences,
                instruction=None,
                tokenizer=self.tokenizer,
                batch_size=kwargs.get("batch_size", 32),
                normalize=True,
                convert_to_numpy=True,
            )

    mteb_model = MTEBModel(model_path)

    # Select tasks
    if tasks is not None:
        task_list = [t.strip() for t in tasks.split(",")]
    else:
        task_list = None  # Run all tasks

    # Run evaluation
    evaluation = MTEB(tasks=task_list)
    results = evaluation.run(mteb_model)

    return results


def main():
    args = parse_args()

    print("\n" + "="*80)
    print("EMBEDDING MODEL EVALUATION")
    print("="*80 + "\n")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = InstructionAwareEmbeddingModel.from_pretrained(args.model_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        # Fallback
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully")

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    # Custom evaluation
    if args.eval_data:
        print(f"\nLoading evaluation data from {args.eval_data}...")
        eval_data = load_jsonl(args.eval_data)
        print(f"  Loaded {len(eval_data)} samples")

        # Determine task type
        task_type = eval_data[0].get("task_type", "retrieval")
        print(f"  Task type: {task_type}")

        # Run evaluation based on task type
        if task_type == "retrieval":
            results = evaluate_retrieval(model, tokenizer, eval_data, args.batch_size)
        elif task_type == "sts":
            results = evaluate_sts(model, tokenizer, eval_data, args.batch_size)
        elif task_type == "classification":
            results = evaluate_classification(model, tokenizer, eval_data, args.batch_size)
        else:
            print(f"Unknown task type: {task_type}")
            results = {}

        all_results[task_type] = results

        # Print results
        print(f"\n{task_type.upper()} Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")

    # MTEB evaluation
    if args.use_mteb:
        mteb_results = run_mteb_evaluation(args.model_path, args.mteb_tasks)
        if mteb_results:
            all_results["mteb"] = mteb_results

    # Save results
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
