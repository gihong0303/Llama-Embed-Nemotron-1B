"""
Model Merging (Model Soup)

Based on Llama-Embed-Nemotron paper:
- Train multiple models with different hyperparameters/data mixes
- Merge them by averaging parameters (uniform or weighted)
- Results in better generalization without additional inference cost

Reference:
- Model Soups: https://arxiv.org/abs/2203.05482
- Paper Table 7: Shows improvements from merging 6 models
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict
from pathlib import Path
import os
from tqdm import tqdm

from ..models.embedding_model import InstructionAwareEmbeddingModel


class ModelMerger:
    """
    Merge multiple fine-tuned models into one.

    Merging strategies:
    - uniform: Simple average of all models (default, used in paper)
    - weighted: Weighted average based on validation performance

    Args:
        model_paths: List of paths to models to merge
        weights: Optional weights for each model (for weighted averaging)
        strategy: "uniform" or "weighted"
    """

    def __init__(
        self,
        model_paths: List[str],
        weights: Optional[List[float]] = None,
        strategy: str = "uniform",
    ):
        self.model_paths = model_paths
        self.strategy = strategy

        # Validate
        if len(model_paths) == 0:
            raise ValueError("Must provide at least one model path")

        # Set weights
        if weights is None or strategy == "uniform":
            # Uniform weights
            self.weights = [1.0 / len(model_paths)] * len(model_paths)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]

        if len(self.weights) != len(model_paths):
            raise ValueError("Number of weights must match number of models")

        print(f"ModelMerger initialized with {len(model_paths)} models")
        print(f"  Strategy: {strategy}")
        for i, (path, weight) in enumerate(zip(model_paths, self.weights)):
            print(f"  Model {i+1}: {path} (weight: {weight:.4f})")

    def merge(self, output_path: str) -> InstructionAwareEmbeddingModel:
        """
        Merge models and save to output_path.

        Args:
            output_path: Path to save merged model

        Returns:
            merged_model: Merged model
        """
        print("\nMerging models...")

        # Load all models
        models = []
        for path in tqdm(self.model_paths, desc="Loading models"):
            model = InstructionAwareEmbeddingModel.from_pretrained(path)
            model.eval()
            models.append(model)

        # Initialize merged model from first model
        merged_model = InstructionAwareEmbeddingModel.from_pretrained(self.model_paths[0])

        # Merge parameters
        print("Merging parameters...")
        merged_state_dict = {}

        # Get parameter names from first model
        first_state_dict = models[0].state_dict()

        for param_name in tqdm(first_state_dict.keys(), desc="Merging"):
            # Collect parameter from all models
            param_values = []
            for model in models:
                param = model.state_dict()[param_name]
                param_values.append(param)

            # Weighted average
            merged_param = torch.zeros_like(param_values[0])
            for param, weight in zip(param_values, self.weights):
                merged_param += param * weight

            merged_state_dict[param_name] = merged_param

        # Load merged parameters
        merged_model.load_state_dict(merged_state_dict)

        # Save
        print(f"Saving merged model to {output_path}...")
        os.makedirs(output_path, exist_ok=True)
        merged_model.save_pretrained(output_path)

        print("Merging complete!")

        return merged_model

    @staticmethod
    def compute_optimal_weights(
        model_paths: List[str],
        eval_losses: List[float],
        temperature: float = 1.0,
    ) -> List[float]:
        """
        Compute optimal weights based on validation losses.

        Uses softmax with temperature to convert losses to weights.
        Lower loss = higher weight.

        Args:
            model_paths: List of model paths
            eval_losses: Validation losses for each model
            temperature: Softmax temperature (higher = more uniform)

        Returns:
            weights: Normalized weights
        """
        import numpy as np

        # Invert losses (lower is better)
        inverted_losses = [1.0 / (loss + 1e-8) for loss in eval_losses]

        # Apply softmax with temperature
        exp_values = np.exp(np.array(inverted_losses) / temperature)
        weights = exp_values / exp_values.sum()

        weights = weights.tolist()

        print("Computed optimal weights:")
        for path, loss, weight in zip(model_paths, eval_losses, weights):
            print(f"  {path}: loss={loss:.4f}, weight={weight:.4f}")

        return weights


def merge_models_uniform(
    model_paths: List[str],
    output_path: str,
) -> InstructionAwareEmbeddingModel:
    """
    Merge models with uniform averaging (simplest approach, used in paper).

    Args:
        model_paths: List of model checkpoint paths
        output_path: Where to save merged model

    Returns:
        merged_model: Merged model
    """
    merger = ModelMerger(model_paths, strategy="uniform")
    merged_model = merger.merge(output_path)
    return merged_model


def merge_models_weighted(
    model_paths: List[str],
    eval_losses: List[float],
    output_path: str,
    temperature: float = 1.0,
) -> InstructionAwareEmbeddingModel:
    """
    Merge models with weighted averaging based on validation performance.

    Args:
        model_paths: List of model checkpoint paths
        eval_losses: Validation loss for each model
        output_path: Where to save merged model
        temperature: Softmax temperature for weight computation

    Returns:
        merged_model: Merged model
    """
    # Compute optimal weights
    weights = ModelMerger.compute_optimal_weights(model_paths, eval_losses, temperature)

    # Merge
    merger = ModelMerger(model_paths, weights=weights, strategy="weighted")
    merged_model = merger.merge(output_path)

    return merged_model


def create_ensemble_predictions(
    models: List[InstructionAwareEmbeddingModel],
    texts: List[str],
    instruction: Optional[str] = None,
    weights: Optional[List[float]] = None,
    tokenizer = None,
) -> torch.Tensor:
    """
    Create ensemble predictions by averaging embeddings from multiple models.

    Note: This is for prediction time. For parameter merging (what the paper uses),
    use merge_models_uniform or merge_models_weighted instead.

    Args:
        models: List of models
        texts: List of texts to encode
        instruction: Task instruction
        weights: Optional weights for each model
        tokenizer: Tokenizer (will load default if None)

    Returns:
        embeddings: Averaged embeddings [num_texts, hidden_size]
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    all_embeddings = []

    for model, weight in zip(models, weights):
        embeds = model.encode(texts, instruction=instruction, tokenizer=tokenizer)
        all_embeddings.append(embeds * weight)

    # Average
    ensemble_embeddings = torch.stack(all_embeddings).sum(dim=0)

    return ensemble_embeddings


# Example usage
if __name__ == "__main__":
    # Example: Merge 6 models (as in paper)
    model_paths = [
        "./outputs/run1/stage2/best_model",
        "./outputs/run2/stage2/best_model",
        "./outputs/run3/stage2/best_model",
        "./outputs/run4/stage2/best_model",
        "./outputs/run5/stage2/best_model",
        "./outputs/run6/stage2/best_model",
    ]

    output_path = "./outputs/merged_model"

    # Uniform merging (used in paper)
    merged_model = merge_models_uniform(model_paths, output_path)

    print(f"Merged model saved to {output_path}")

    # Alternative: Weighted merging
    # eval_losses = [0.45, 0.48, 0.44, 0.47, 0.46, 0.43]  # Example losses
    # merged_model = merge_models_weighted(model_paths, eval_losses, output_path)
