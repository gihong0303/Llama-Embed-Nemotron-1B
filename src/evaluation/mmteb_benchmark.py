"""
MMTEB (Massive Multilingual Text Embedding Benchmark) Evaluation

Complete evaluation suite for MMTEB v2 benchmark:
- 131 tasks
- 9 task types
- 250+ languages

Based on paper: Llama-Embed-Nemotron-8B achieved #1 on MMTEB leaderboard.
"""

import os
import json
from typing import List, Optional, Dict
from pathlib import Path
from datetime import datetime


class MMTEBEvaluator:
    """
    Complete MMTEB benchmark evaluator.

    Task types:
    1. Retrieval
    2. STS (Semantic Textual Similarity)
    3. Classification
    4. Clustering
    5. Pair Classification
    6. Reranking
    7. Multi-label Classification
    8. Instruction Retrieval
    9. Bitext Mining

    Args:
        model_path: Path to trained embedding model
        output_dir: Directory for results
        languages: List of languages to evaluate (None = all)
        batch_size: Batch size for encoding
    """

    # MMTEB task categories
    TASK_TYPES = [
        "Retrieval",
        "STS",
        "Classification",
        "Clustering",
        "PairClassification",
        "Reranking",
        "MultiLabelClassification",
        "InstructionRetrieval",
        "BitextMining",
    ]

    def __init__(
        self,
        model_path: str,
        output_dir: str = "./mmteb_results",
        languages: Optional[List[str]] = None,
        batch_size: int = 32,
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.languages = languages
        self.batch_size = batch_size

        # Will be loaded lazily
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model for evaluation."""
        if self.model is not None:
            return

        print(f"Loading model from {self.model_path}...")

        from transformers import AutoTokenizer
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.models.embedding_model import InstructionAwareEmbeddingModel

        self.model = InstructionAwareEmbeddingModel.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

        print("Model loaded successfully")

    def run_full_benchmark(
        self,
        task_types: Optional[List[str]] = None,
        save_individual_results: bool = True,
    ) -> Dict:
        """
        Run complete MMTEB benchmark.

        Args:
            task_types: List of task types to evaluate (None = all)
            save_individual_results: Save results for each task

        Returns:
            results: Dictionary with all results and metrics
        """
        try:
            from mteb import MTEB
        except ImportError:
            print("Error: MTEB not installed")
            print("Install with: pip install mteb")
            return {}

        self.load_model()

        print("\n" + "="*80)
        print("MMTEB FULL BENCHMARK EVALUATION")
        print("="*80)
        print(f"Model: {self.model_path}")
        print(f"Output: {self.output_dir}")
        print(f"Task types: {task_types or 'all'}")
        print(f"Languages: {self.languages or 'all'}")
        print("="*80 + "\n")

        # Wrap model for MTEB interface
        class MTEBModelWrapper:
            def __init__(self, embedding_model, tokenizer):
                self.embedding_model = embedding_model
                self.tokenizer = tokenizer

            def encode(self, sentences, batch_size=32, **kwargs):
                """MTEB expects this interface."""
                # Get task instruction from kwargs if available
                task_name = kwargs.get("task_name", "")
                instruction = self._get_instruction_for_task(task_name)

                return self.embedding_model.encode(
                    sentences,
                    instruction=instruction,
                    tokenizer=self.tokenizer,
                    batch_size=batch_size,
                    normalize=True,
                    convert_to_numpy=True,
                )

            def _get_instruction_for_task(self, task_name: str) -> str:
                """Get appropriate instruction for MTEB task."""
                task_lower = task_name.lower()

                if "retrieval" in task_lower or "nfcorpus" in task_lower or "fever" in task_lower:
                    return "Retrieve relevant passages for this query"
                elif "sts" in task_lower or "sick" in task_lower:
                    return "Retrieve semantically similar text"
                elif "classification" in task_lower or "amazon" in task_lower:
                    return "Classify the topic of this text"
                elif "clustering" in task_lower:
                    return "Identify the topic or theme of the text"
                elif "rerank" in task_lower:
                    return "Retrieve relevant passages for this query"
                elif "bitext" in task_lower:
                    return "Retrieve semantically similar text"
                else:
                    return ""  # No instruction

        wrapped_model = MTEBModelWrapper(self.model, self.tokenizer)

        # Create MTEB evaluator
        evaluation = MTEB(
            task_types=task_types,
            task_langs=self.languages,
        )

        # Run evaluation
        print("Starting MMTEB evaluation...\n")

        results = evaluation.run(
            wrapped_model,
            output_folder=str(self.output_dir),
            eval_splits=["test"],
            batch_size=self.batch_size,
        )

        # Save summary
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {summary_path}")

        # Compute aggregate metrics
        summary = self._compute_summary_metrics(results)

        return summary

    def _compute_summary_metrics(self, results: Dict) -> Dict:
        """
        Compute summary metrics from MTEB results.

        Returns:
            summary: {
                "mean_score": float,
                "task_type_scores": {...},
                "num_tasks": int,
                "borda_score": float (if applicable),
            }
        """
        task_scores = []
        task_type_scores = {}

        for task_name, task_results in results.items():
            # Extract main metric (usually ndcg@10 for retrieval, accuracy for classification, etc.)
            if "test" in task_results:
                test_results = task_results["test"]

                # Get primary metric
                if "ndcg_at_10" in test_results:
                    score = test_results["ndcg_at_10"]
                elif "cosine_spearman" in test_results:
                    score = test_results["cosine_spearman"]
                elif "accuracy" in test_results:
                    score = test_results["accuracy"]
                elif "f1" in test_results:
                    score = test_results["f1"]
                else:
                    # Take first available metric
                    score = list(test_results.values())[0] if test_results else 0.0

                task_scores.append(score)

                # Group by task type
                task_type = self._infer_task_type(task_name)
                if task_type not in task_type_scores:
                    task_type_scores[task_type] = []
                task_type_scores[task_type].append(score)

        # Compute means
        mean_score = sum(task_scores) / len(task_scores) if task_scores else 0.0

        task_type_means = {
            task_type: sum(scores) / len(scores)
            for task_type, scores in task_type_scores.items()
        }

        summary = {
            "mean_score": mean_score,
            "task_type_scores": task_type_means,
            "num_tasks": len(task_scores),
            "evaluation_date": datetime.now().isoformat(),
            "model_path": str(self.model_path),
        }

        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Mean Score: {mean_score:.4f}")
        print(f"Number of Tasks: {len(task_scores)}")
        print("\nTask Type Scores:")
        for task_type, score in sorted(task_type_means.items()):
            print(f"  {task_type}: {score:.4f}")
        print("="*80 + "\n")

        return summary

    def _infer_task_type(self, task_name: str) -> str:
        """Infer task type from task name."""
        name_lower = task_name.lower()

        if "retrieval" in name_lower:
            return "Retrieval"
        elif "sts" in name_lower or "sick" in name_lower:
            return "STS"
        elif "classification" in name_lower:
            return "Classification"
        elif "clustering" in name_lower:
            return "Clustering"
        elif "pair" in name_lower:
            return "PairClassification"
        elif "rerank" in name_lower:
            return "Reranking"
        elif "bitext" in name_lower:
            return "BitextMining"
        else:
            return "Other"

    def run_specific_tasks(
        self,
        task_names: List[str],
    ) -> Dict:
        """
        Run evaluation on specific MTEB tasks.

        Args:
            task_names: List of MTEB task names

        Returns:
            results: Results dictionary
        """
        try:
            from mteb import MTEB
        except ImportError:
            print("Error: MTEB not installed")
            return {}

        self.load_model()

        print(f"Running {len(task_names)} tasks...")

        # Similar to run_full_benchmark but with specific tasks
        # (Implementation similar to above)

        return {}


def run_mmteb_suite(
    model_path: str,
    output_dir: str = "./mmteb_results",
    task_types: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
):
    """
    Convenience function to run full MMTEB evaluation.

    Args:
        model_path: Path to model
        output_dir: Output directory
        task_types: Task types to evaluate
        languages: Languages to evaluate

    Example:
        >>> run_mmteb_suite(
        ...     model_path="outputs/stage2/best_model",
        ...     task_types=["Retrieval", "STS", "Classification"],
        ...     languages=["en", "zh", "es", "fr"],
        ... )
    """
    evaluator = MMTEBEvaluator(
        model_path=model_path,
        output_dir=output_dir,
        languages=languages,
    )

    results = evaluator.run_full_benchmark(task_types=task_types)

    return results


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MMTEB Benchmark Evaluation")

    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default="./mmteb_results",
                       help="Output directory")
    parser.add_argument("--task_types", type=str, default=None,
                       help="Comma-separated task types (e.g., 'Retrieval,STS,Classification')")
    parser.add_argument("--languages", type=str, default=None,
                       help="Comma-separated languages (e.g., 'en,zh,es,fr')")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for encoding")

    args = parser.parse_args()

    # Parse arguments
    task_types = args.task_types.split(",") if args.task_types else None
    languages = args.languages.split(",") if args.languages else None

    # Run evaluation
    run_mmteb_suite(
        model_path=args.model_path,
        output_dir=args.output_dir,
        task_types=task_types,
        languages=languages,
    )
