"""
Generate Synthetic Training Data

Use LLMs to generate synthetic query-document pairs for training.

Usage:
    # Generate from corpus
    python scripts/generate_synthetic_data.py \
        --corpus data/corpus.txt \
        --output data/synthetic_retrieval.jsonl \
        --model_name meta-llama/Llama-3.2-1B \
        --num_queries_per_doc 2

    # Generate with multiple LLMs
    python scripts/generate_synthetic_data.py \
        --corpus data/corpus.txt \
        --output data/synthetic_retrieval.jsonl \
        --multi_llm \
        --model_names "meta-llama/Llama-3.2-1B,meta-llama/Llama-3.2-3B"
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from src.data.synthetic_generation import (
    SyntheticDataGenerator,
    MultiLLMSyntheticGenerator,
    load_corpus_from_file,
    save_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Synthetic Training Data")

    # Input
    parser.add_argument("--corpus", type=str, required=True,
                       help="Path to corpus file (one document per line)")
    parser.add_argument("--task_type", type=str, default="retrieval",
                       choices=["retrieval", "classification", "paraphrase"],
                       help="Type of synthetic data to generate")

    # Output
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for generated data (JSONL)")

    # LLM settings
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B",
                       help="LLM model name for generation")
    parser.add_argument("--multi_llm", action="store_true",
                       help="Use multiple LLMs for diversity")
    parser.add_argument("--model_names", type=str, default=None,
                       help="Comma-separated list of model names for multi-LLM generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Max tokens to generate")

    # Generation settings
    parser.add_argument("--num_queries_per_doc", type=int, default=1,
                       help="Number of queries to generate per document")
    parser.add_argument("--max_docs", type=int, default=None,
                       help="Maximum number of documents to process (for testing)")

    # Classification specific
    parser.add_argument("--labels", type=str, default=None,
                       help="Comma-separated list of labels for classification")
    parser.add_argument("--num_examples_per_label", type=int, default=10,
                       help="Number of examples per label")

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*80)
    print("SYNTHETIC DATA GENERATION")
    print("="*80 + "\n")

    # Load corpus
    if args.task_type in ["retrieval", "paraphrase"]:
        print(f"Loading corpus from {args.corpus}...")
        corpus = load_corpus_from_file(args.corpus)

        if args.max_docs:
            corpus = corpus[:args.max_docs]

        print(f"  Loaded {len(corpus)} documents")

    # Create generator(s)
    if args.multi_llm:
        if args.model_names is None:
            raise ValueError("Must provide --model_names when using --multi_llm")

        model_names = [m.strip() for m in args.model_names.split(",")]
        print(f"Using multi-LLM generation with {len(model_names)} models:")
        for name in model_names:
            print(f"  - {name}")

        generator = MultiLLMSyntheticGenerator(
            model_names=model_names,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        print(f"Using single LLM: {args.model_name}")
        generator = SyntheticDataGenerator(
            model_name=args.model_name,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )

    # Generate based on task type
    if args.task_type == "retrieval":
        print(f"\nGenerating retrieval data ({args.num_queries_per_doc} queries per document)...")

        if args.multi_llm:
            dataset = generator.generate_retrieval_dataset(
                documents=corpus,
                queries_per_doc=args.num_queries_per_doc,
                show_progress=True,
            )
        else:
            dataset = generator.generate_retrieval_dataset(
                documents=corpus,
                queries_per_doc=args.num_queries_per_doc,
                show_progress=True,
            )

    elif args.task_type == "classification":
        if args.labels is None:
            raise ValueError("Must provide --labels for classification task")

        labels = [l.strip() for l in args.labels.split(",")]
        print(f"\nGenerating classification data for {len(labels)} labels...")
        print(f"  Labels: {labels}")

        if args.multi_llm:
            examples = generator.generate_classification_examples(
                labels=labels,
                num_examples_per_label=args.num_examples_per_label,
                show_progress=True,
            )
        else:
            examples = generator.generate_classification_examples(
                labels=labels,
                num_examples_per_label=args.num_examples_per_label,
                show_progress=True,
            )

        # Convert to standard format
        dataset = []
        for ex in examples:
            dataset.append({
                "query": ex["text"],
                "positive": ex["label"],
                "negatives": [],  # Will be filled by hard negative mining
                "task_type": "classification",
            })

    elif args.task_type == "paraphrase":
        print(f"\nGenerating paraphrases...")

        paraphrases = generator.augment_with_paraphrases(
            texts=corpus,
            num_paraphrases=args.num_queries_per_doc,
            show_progress=True,
        )

        # Convert to STS format
        dataset = []
        for original, paraphrase_list in zip(corpus, paraphrases):
            for paraphrase in paraphrase_list:
                if paraphrase:
                    dataset.append({
                        "query": original,
                        "positive": paraphrase,
                        "negatives": [],
                        "task_type": "sts",
                    })

    # Save
    print(f"\nSaving to {args.output}...")
    save_dataset(dataset, args.output)

    print("\nGeneration complete!")
    print(f"  Generated {len(dataset)} samples")


if __name__ == "__main__":
    main()
