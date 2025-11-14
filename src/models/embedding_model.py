"""
Instruction-Aware Embedding Model

Wrapper around BiDirectionalLlamaModel that adds:
1. Instruction-aware input formatting
2. Mean pooling over sequence dimension
3. L2 normalization of embeddings

Based on Llama-Embed-Nemotron paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union
from transformers import AutoTokenizer, PreTrainedTokenizer
from .bidirectional_llama import BiDirectionalLlamaModel


class InstructionAwareEmbeddingModel(nn.Module):
    """
    Instruction-aware text embedding model.

    This model wraps BiDirectionalLlamaModel and adds:
    1. Instruction formatting: "Instruct: {instruction}\nQuery: {text}"
    2. Mean pooling over the sequence dimension
    3. Optional L2 normalization

    Args:
        model_name_or_path: Path to pre-trained Llama model
        hidden_size: Hidden dimension (default: 2048 for Llama-3.2-1B)
        normalize_embeddings: Whether to L2 normalize embeddings (default: True)

    Example:
        >>> model = InstructionAwareEmbeddingModel("meta-llama/Llama-3.2-1B")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        >>>
        >>> # Encode with instruction
        >>> instruction = "Retrieve relevant passages for question answering"
        >>> texts = ["What is the capital of France?", "Paris is the capital of France."]
        >>> embeddings = model.encode(texts, instruction, tokenizer)
    """

    # Default instructions for different task types (from paper)
    INSTRUCTIONS = {
        "retrieval_query": "Retrieve relevant passages for this query",
        "retrieval_passage": "",  # No instruction for passages
        "sts": "Retrieve semantically similar text",
        "classification": "Classify the topic of this text",
        "clustering": "Identify the topic or theme of the text",
        "qa": "Retrieve relevant passages for question answering",
    }

    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Llama-3.2-1B",
        hidden_size: int = 2048,
        normalize_embeddings: bool = True,
        max_length: int = 512,
    ):
        super().__init__()

        self.model_name = model_name_or_path
        self.hidden_size = hidden_size
        self.normalize_embeddings = normalize_embeddings
        self.max_length = max_length

        # Load bi-directional Llama model
        print(f"Loading bi-directional Llama model from {model_name_or_path}...")
        self.encoder = BiDirectionalLlamaModel.from_pretrained(model_name_or_path)

        # Ensure all parameters are trainable
        for param in self.encoder.parameters():
            param.requires_grad = True

        print(f"Model loaded. Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def mean_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mean pooling over sequence dimension, respecting attention mask.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]

        Returns:
            pooled: [batch_size, hidden_size]
        """
        # Expand attention mask to match hidden_states shape
        # attention_mask: [batch_size, seq_len, 1]
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

        # Sum of hidden states, weighted by attention mask
        sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, dim=1)

        # Sum of attention mask (to get number of non-padding tokens)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)

        # Mean pooling
        mean_embeddings = sum_embeddings / sum_mask

        return mean_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        normalize: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Forward pass to generate embeddings.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            normalize: Whether to L2 normalize (overrides self.normalize_embeddings)

        Returns:
            embeddings: [batch_size, hidden_size]
        """
        # Get hidden states from encoder
        hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )  # [batch_size, seq_len, hidden_size]

        # Mean pooling
        embeddings = self.mean_pooling(hidden_states, attention_mask)

        # L2 normalization
        if normalize is None:
            normalize = self.normalize_embeddings

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        instruction: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        batch_size: int = 32,
        max_length: Optional[int] = None,
        normalize: bool = True,
        convert_to_numpy: bool = False,
        show_progress: bool = False,
    ) -> Union[torch.Tensor, 'numpy.ndarray']:
        """
        Encode texts into embeddings.

        Args:
            texts: Single text or list of texts to encode
            instruction: Task instruction to prepend (e.g., "Retrieve relevant passages")
            tokenizer: Tokenizer to use (will load default if not provided)
            batch_size: Batch size for encoding
            max_length: Maximum sequence length (uses self.max_length if None)
            normalize: Whether to L2 normalize embeddings
            convert_to_numpy: Convert output to numpy array
            show_progress: Show progress bar

        Returns:
            embeddings: [num_texts, hidden_size]
        """
        self.eval()

        if isinstance(texts, str):
            texts = [texts]

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        if max_length is None:
            max_length = self.max_length

        # Format texts with instruction
        formatted_texts = self._format_texts_with_instruction(texts, instruction)

        # Encode in batches
        all_embeddings = []

        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(formatted_texts), batch_size), desc="Encoding")
        else:
            iterator = range(0, len(formatted_texts), batch_size)

        for i in iterator:
            batch_texts = formatted_texts[i:i + batch_size]

            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            # Move to model device
            input_ids = encoded["input_ids"].to(self.encoder.embed_tokens.weight.device)
            attention_mask = encoded["attention_mask"].to(self.encoder.embed_tokens.weight.device)

            # Generate embeddings
            batch_embeddings = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                normalize=normalize,
            )

            all_embeddings.append(batch_embeddings.cpu())

        # Concatenate all embeddings
        embeddings = torch.cat(all_embeddings, dim=0)

        if convert_to_numpy:
            embeddings = embeddings.numpy()

        return embeddings

    def _format_texts_with_instruction(
        self,
        texts: List[str],
        instruction: Optional[str] = None
    ) -> List[str]:
        """
        Format texts with instruction prefix.

        Format: "Instruct: {instruction}\nQuery: {text}"

        Args:
            texts: List of texts
            instruction: Task instruction (can be None for passages)

        Returns:
            formatted_texts: List of formatted texts
        """
        if instruction is None or instruction == "":
            # No instruction (e.g., for passages in retrieval)
            return texts

        formatted = []
        for text in texts:
            formatted_text = f"Instruct: {instruction}\nQuery: {text}"
            formatted.append(formatted_text)

        return formatted

    def get_task_instruction(self, task_type: str) -> str:
        """
        Get default instruction for a task type.

        Args:
            task_type: One of ["retrieval_query", "retrieval_passage", "sts", "classification", "clustering", "qa"]

        Returns:
            instruction: Default instruction for the task
        """
        return self.INSTRUCTIONS.get(task_type, "")

    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Save encoder
        self.encoder.save_pretrained(save_directory)

        # Save config
        config = {
            "model_name": self.model_name,
            "hidden_size": self.hidden_size,
            "normalize_embeddings": self.normalize_embeddings,
            "max_length": self.max_length,
        }

        import json
        with open(os.path.join(save_directory, "embedding_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, model_path: str):
        """Load model from directory."""
        import json
        import os

        # Load config
        config_path = os.path.join(model_path, "embedding_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)

            model = cls(
                model_name_or_path=model_path,
                hidden_size=config.get("hidden_size", 2048),
                normalize_embeddings=config.get("normalize_embeddings", True),
                max_length=config.get("max_length", 512),
            )
        else:
            # Fallback to default initialization
            model = cls(model_name_or_path=model_path)

        print(f"Model loaded from {model_path}")
        return model


def create_embedding_model(
    model_name: str = "meta-llama/Llama-3.2-1B",
    normalize: bool = True,
    max_length: int = 512,
) -> InstructionAwareEmbeddingModel:
    """
    Factory function to create an embedding model.

    Args:
        model_name: Hugging Face model name or path
        normalize: Whether to L2 normalize embeddings
        max_length: Maximum sequence length

    Returns:
        Embedding model
    """
    # Get hidden size based on model
    if "1B" in model_name or "1b" in model_name:
        hidden_size = 2048
    elif "3B" in model_name or "3b" in model_name:
        hidden_size = 3072
    elif "8B" in model_name or "8b" in model_name:
        hidden_size = 4096
    else:
        # Default
        hidden_size = 2048

    model = InstructionAwareEmbeddingModel(
        model_name_or_path=model_name,
        hidden_size=hidden_size,
        normalize_embeddings=normalize,
        max_length=max_length,
    )

    return model
