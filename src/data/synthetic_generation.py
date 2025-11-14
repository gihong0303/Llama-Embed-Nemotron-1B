"""
Synthetic Data Generation (SDG) for Text Embedding

Based on Llama-Embed-Nemotron paper:
- Two strategies:
  1. From scratch: LLM generates (query, positive, negatives) triplets
  2. From corpus: Given a document, LLM generates relevant queries

- Use multiple LLMs for diversity (mix of outputs is better than single LLM)
- Support for retrieval, STS, classification, bitext tasks
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Union
import random
from tqdm import tqdm
import json


class SyntheticDataGenerator:
    """
    Generate synthetic training data using LLMs.

    Args:
        model_name: LLM model name (e.g., "meta-llama/Llama-3.2-1B")
        device: Device to run on
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        device: Optional[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        print(f"Loading LLM for SDG: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device == "cuda" else None,
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"LLM loaded on {device}")

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Max tokens to generate (uses self.max_new_tokens if None)

        Returns:
            generated_text: Generated text
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode and remove prompt
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_text[len(prompt):].strip()

        return generated_text

    def generate_query_from_document(
        self,
        document: str,
        task_type: str = "retrieval",
        num_queries: int = 1,
    ) -> List[str]:
        """
        Generate queries for a given document.

        Args:
            document: Source document
            task_type: Type of task ("retrieval", "qa", etc.)
            num_queries: Number of queries to generate

        Returns:
            queries: List of generated queries
        """
        if task_type == "retrieval" or task_type == "qa":
            prompt = f"""Given the following passage, generate {num_queries} relevant question(s) that this passage would answer.

Passage: {document}

Generate natural, diverse questions. Output only the questions, one per line.

Questions:"""
        else:
            prompt = f"""Given the following text, generate {num_queries} semantically similar text(s).

Text: {document}

Generate natural, diverse texts. Output only the texts, one per line.

Similar texts:"""

        generated = self.generate(prompt)

        # Parse generated queries (split by newlines)
        queries = [q.strip() for q in generated.split("\n") if q.strip()]
        queries = [q.lstrip("0123456789.-) ") for q in queries]  # Remove numbering

        return queries[:num_queries]

    def generate_triplet_from_scratch(
        self,
        topic: Optional[str] = None,
        task_type: str = "retrieval",
    ) -> Dict[str, Union[str, List[str]]]:
        """
        Generate (query, positive, negatives) triplet from scratch.

        Args:
            topic: Optional topic to guide generation
            task_type: Type of task

        Returns:
            triplet: {"query": str, "positive": str, "negatives": List[str]}
        """
        if topic:
            topic_instruction = f" about {topic}"
        else:
            topic_instruction = ""

        prompt = f"""Generate a training example for a text retrieval system{topic_instruction}.

Output format:
Query: <a question or search query>
Positive: <a relevant passage that answers the query>
Negative 1: <an irrelevant or partially relevant passage>
Negative 2: <another irrelevant or partially relevant passage>

Example:"""

        generated = self.generate(prompt, max_new_tokens=400)

        # Parse output
        lines = generated.split("\n")
        triplet = {
            "query": "",
            "positive": "",
            "negatives": [],
        }

        for line in lines:
            line = line.strip()
            if line.startswith("Query:"):
                triplet["query"] = line.replace("Query:", "").strip()
            elif line.startswith("Positive:"):
                triplet["positive"] = line.replace("Positive:", "").strip()
            elif line.startswith("Negative"):
                neg_text = line.split(":", 1)[1].strip() if ":" in line else ""
                if neg_text:
                    triplet["negatives"].append(neg_text)

        return triplet

    def generate_classification_examples(
        self,
        labels: List[str],
        num_examples_per_label: int = 10,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Generate classification examples for given labels.

        Args:
            labels: List of class labels
            num_examples_per_label: Number of examples to generate per label
            show_progress: Show progress bar

        Returns:
            examples: List of {"text": str, "label": str}
        """
        examples = []

        iterator = tqdm(labels, desc="Generating classification data") if show_progress else labels

        for label in iterator:
            prompt = f"""Generate {num_examples_per_label} example texts that belong to the category "{label}".

Output format (one text per line):
1. <example text 1>
2. <example text 2>
...

Examples:"""

            generated = self.generate(prompt, max_new_tokens=500)

            # Parse
            texts = [t.strip() for t in generated.split("\n") if t.strip()]
            texts = [t.lstrip("0123456789.-) ") for t in texts]

            for text in texts[:num_examples_per_label]:
                if text:
                    examples.append({
                        "text": text,
                        "label": label,
                    })

        return examples

    def generate_retrieval_dataset(
        self,
        documents: List[str],
        queries_per_doc: int = 1,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Generate retrieval dataset from document corpus.

        Args:
            documents: List of documents
            queries_per_doc: Number of queries to generate per document
            show_progress: Show progress bar

        Returns:
            dataset: List of {"query": str, "positive": str}
        """
        dataset = []

        iterator = tqdm(documents, desc="Generating queries") if show_progress else documents

        for doc in iterator:
            queries = self.generate_query_from_document(doc, num_queries=queries_per_doc)

            for query in queries:
                if query:
                    dataset.append({
                        "query": query,
                        "positive": doc,
                    })

        return dataset

    def augment_with_paraphrases(
        self,
        texts: List[str],
        num_paraphrases: int = 1,
        show_progress: bool = True,
    ) -> List[List[str]]:
        """
        Generate paraphrases for texts (useful for STS).

        Args:
            texts: List of texts to paraphrase
            num_paraphrases: Number of paraphrases per text
            show_progress: Show progress bar

        Returns:
            paraphrases: List of lists, where paraphrases[i] contains paraphrases of texts[i]
        """
        all_paraphrases = []

        iterator = tqdm(texts, desc="Generating paraphrases") if show_progress else texts

        for text in iterator:
            prompt = f"""Generate {num_paraphrases} paraphrase(s) of the following text. Keep the meaning the same but use different words and sentence structure.

Original: {text}

Paraphrases (one per line):"""

            generated = self.generate(prompt)

            paraphrases = [p.strip() for p in generated.split("\n") if p.strip()]
            paraphrases = [p.lstrip("0123456789.-) ") for p in paraphrases]

            all_paraphrases.append(paraphrases[:num_paraphrases])

        return all_paraphrases


class MultiLLMSyntheticGenerator:
    """
    Generate synthetic data using multiple LLMs for diversity.

    Based on paper finding: mixing outputs from multiple LLMs is better than using a single LLM.

    Args:
        model_names: List of LLM model names
        weights: Sampling weights for each LLM (optional)
    """

    def __init__(
        self,
        model_names: List[str],
        weights: Optional[List[float]] = None,
        **generator_kwargs
    ):
        self.model_names = model_names
        self.generators = []

        # Create generators (but don't load all at once to save memory)
        self.generator_kwargs = generator_kwargs

        # Normalize weights
        if weights is None:
            weights = [1.0] * len(model_names)
        total = sum(weights)
        self.weights = [w / total for w in weights]

        print(f"MultiLLMSyntheticGenerator created with {len(model_names)} models:")
        for name, weight in zip(model_names, self.weights):
            print(f"  - {name} (weight: {weight:.2f})")

    def generate_retrieval_dataset(
        self,
        documents: List[str],
        queries_per_doc: int = 1,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Generate retrieval dataset using multiple LLMs.

        Documents are distributed among LLMs based on weights.
        """
        dataset = []

        # Split documents among LLMs based on weights
        num_docs_per_model = [int(len(documents) * w) for w in self.weights]

        # Adjust last to ensure we use all documents
        num_docs_per_model[-1] = len(documents) - sum(num_docs_per_model[:-1])

        start_idx = 0
        for model_name, num_docs in zip(self.model_names, num_docs_per_model):
            if num_docs == 0:
                continue

            print(f"\nGenerating with {model_name} ({num_docs} documents)...")

            # Create generator
            generator = SyntheticDataGenerator(model_name, **self.generator_kwargs)

            # Generate for subset
            doc_subset = documents[start_idx:start_idx + num_docs]
            subset_data = generator.generate_retrieval_dataset(
                doc_subset,
                queries_per_doc=queries_per_doc,
                show_progress=show_progress,
            )

            dataset.extend(subset_data)
            start_idx += num_docs

            # Free memory
            del generator
            torch.cuda.empty_cache()

        print(f"\nGenerated {len(dataset)} query-document pairs using {len(self.model_names)} LLMs")

        return dataset

    def generate_classification_examples(
        self,
        labels: List[str],
        num_examples_per_label: int = 10,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Generate classification examples using multiple LLMs.
        """
        all_examples = []

        # Each LLM generates for all labels
        num_per_llm = max(1, num_examples_per_label // len(self.model_names))

        for model_name in self.model_names:
            print(f"\nGenerating with {model_name}...")

            generator = SyntheticDataGenerator(model_name, **self.generator_kwargs)

            examples = generator.generate_classification_examples(
                labels,
                num_examples_per_label=num_per_llm,
                show_progress=show_progress,
            )

            all_examples.extend(examples)

            del generator
            torch.cuda.empty_cache()

        print(f"\nGenerated {len(all_examples)} classification examples using {len(self.model_names)} LLMs")

        return all_examples


def load_corpus_from_file(file_path: str) -> List[str]:
    """Load corpus from text file (one document per line)."""
    with open(file_path, "r", encoding="utf-8") as f:
        corpus = [line.strip() for line in f if line.strip()]
    return corpus


def save_dataset(dataset: List[Dict], output_path: str):
    """Save dataset to JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(dataset)} samples to {output_path}")


# Example usage
if __name__ == "__main__":
    # Single LLM example
    generator = SyntheticDataGenerator("meta-llama/Llama-3.2-1B")

    # Generate from document
    doc = "Paris is the capital and most populous city of France."
    queries = generator.generate_query_from_document(doc, num_queries=3)
    print("Generated queries:")
    for q in queries:
        print(f"  - {q}")

    # Generate triplet from scratch
    triplet = generator.generate_triplet_from_scratch(topic="machine learning")
    print("\nGenerated triplet:")
    print(f"Query: {triplet['query']}")
    print(f"Positive: {triplet['positive']}")
    print(f"Negatives: {triplet['negatives']}")
