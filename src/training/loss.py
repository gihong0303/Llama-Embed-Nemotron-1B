"""
InfoNCE Contrastive Loss for Text Embedding

Based on Llama-Embed-Nemotron paper:
- Temperature τ = 0.02
- Hard negatives only (no in-batch or same-tower negatives)
- Pretrain: 1 hard negative per query
- Finetune: 4 hard negatives per query
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss for contrastive learning.

    Loss formula:
        L(q, d+, D_N) = -log( exp(sim(q, d+)/τ) / Σ exp(sim(q, d_i)/τ) )

    where:
        - q: query embedding
        - d+: positive document embedding
        - D_N: set of negative document embeddings
        - sim: cosine similarity
        - τ: temperature parameter

    Args:
        temperature: Temperature parameter (default: 0.02 from paper)
        use_in_batch_negatives: Whether to use other positives in batch as negatives (default: False)

    Example:
        >>> loss_fn = InfoNCELoss(temperature=0.02)
        >>> query_embeds = torch.randn(32, 768)  # [batch_size, hidden_size]
        >>> pos_embeds = torch.randn(32, 768)    # [batch_size, hidden_size]
        >>> neg_embeds = torch.randn(32, 4, 768)  # [batch_size, num_negatives, hidden_size]
        >>> loss = loss_fn(query_embeds, pos_embeds, neg_embeds)
    """

    def __init__(
        self,
        temperature: float = 0.02,
        use_in_batch_negatives: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.use_in_batch_negatives = use_in_batch_negatives

    def forward(
        self,
        query_embeds: torch.Tensor,
        pos_embeds: torch.Tensor,
        neg_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            query_embeds: Query embeddings [batch_size, hidden_size]
            pos_embeds: Positive document embeddings [batch_size, hidden_size]
            neg_embeds: Negative document embeddings [batch_size, num_negatives, hidden_size]
                       Can be None if only using in-batch negatives

        Returns:
            loss: Scalar loss value
        """
        batch_size = query_embeds.size(0)
        device = query_embeds.device

        # Normalize embeddings (for cosine similarity)
        query_embeds = F.normalize(query_embeds, p=2, dim=1)
        pos_embeds = F.normalize(pos_embeds, p=2, dim=1)

        # Compute positive similarities [batch_size]
        pos_sim = torch.sum(query_embeds * pos_embeds, dim=1) / self.temperature

        # Initialize all similarities with positive similarities
        # We'll concatenate negative similarities to this
        all_sims = pos_sim.unsqueeze(1)  # [batch_size, 1]

        # Add hard negative similarities
        if neg_embeds is not None:
            # neg_embeds: [batch_size, num_negatives, hidden_size]
            num_negatives = neg_embeds.size(1)

            # Normalize negatives
            neg_embeds = F.normalize(neg_embeds, p=2, dim=2)

            # Compute negative similarities [batch_size, num_negatives]
            # query_embeds: [batch_size, hidden_size] -> [batch_size, 1, hidden_size]
            # neg_embeds: [batch_size, num_negatives, hidden_size]
            query_expanded = query_embeds.unsqueeze(1)  # [batch_size, 1, hidden_size]
            neg_sim = torch.sum(query_expanded * neg_embeds, dim=2) / self.temperature

            # Concatenate: [batch_size, 1 + num_negatives]
            all_sims = torch.cat([all_sims, neg_sim], dim=1)

        # Add in-batch negatives (other positives in the batch)
        if self.use_in_batch_negatives and batch_size > 1:
            # Compute all pairwise similarities
            # query_embeds: [batch_size, hidden_size]
            # pos_embeds: [batch_size, hidden_size]
            in_batch_sim = torch.matmul(query_embeds, pos_embeds.t()) / self.temperature
            # in_batch_sim: [batch_size, batch_size]

            # Create mask to exclude the positive pair (diagonal)
            mask = torch.eye(batch_size, device=device).bool()
            in_batch_negatives = in_batch_sim.masked_fill(mask, float('-inf'))

            # Concatenate with existing similarities
            all_sims = torch.cat([all_sims, in_batch_negatives], dim=1)

        # Compute log-softmax
        # The first column (index 0) contains the positive similarities
        log_prob = F.log_softmax(all_sims, dim=1)
        pos_log_prob = log_prob[:, 0]

        # Negative log likelihood
        loss = -pos_log_prob.mean()

        return loss

    def forward_symmetric(
        self,
        embeds_a: torch.Tensor,
        embeds_b: torch.Tensor,
        neg_embeds_a: Optional[torch.Tensor] = None,
        neg_embeds_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute symmetric InfoNCE loss for tasks like STS.

        For semantic textual similarity, we compute loss in both directions:
        - L(a, b, negatives_for_a)
        - L(b, a, negatives_for_b)

        Args:
            embeds_a: Embeddings for text A [batch_size, hidden_size]
            embeds_b: Embeddings for text B [batch_size, hidden_size]
            neg_embeds_a: Negatives for A [batch_size, num_negatives, hidden_size]
            neg_embeds_b: Negatives for B [batch_size, num_negatives, hidden_size]

        Returns:
            loss: Symmetric loss (average of both directions)
        """
        # A -> B
        loss_ab = self.forward(embeds_a, embeds_b, neg_embeds_a)

        # B -> A
        loss_ba = self.forward(embeds_b, embeds_a, neg_embeds_b)

        # Average
        loss = (loss_ab + loss_ba) / 2.0

        return loss


class MultiTaskContrastiveLoss(nn.Module):
    """
    Multi-task contrastive loss for handling different task types.

    Supports:
    - Retrieval: asymmetric (query -> document)
    - STS: symmetric (text_a <-> text_b)
    - Classification: (text -> label)

    Args:
        temperature: Temperature parameter
        use_in_batch_negatives: Whether to use in-batch negatives
    """

    def __init__(
        self,
        temperature: float = 0.02,
        use_in_batch_negatives: bool = False,
    ):
        super().__init__()
        self.infonce = InfoNCELoss(temperature, use_in_batch_negatives)

    def forward(
        self,
        query_embeds: torch.Tensor,
        pos_embeds: torch.Tensor,
        neg_embeds: Optional[torch.Tensor] = None,
        task_type: str = "retrieval",
    ) -> torch.Tensor:
        """
        Compute loss based on task type.

        Args:
            query_embeds: Query/text_a embeddings [batch_size, hidden_size]
            pos_embeds: Positive/text_b embeddings [batch_size, hidden_size]
            neg_embeds: Negative embeddings [batch_size, num_negatives, hidden_size]
            task_type: One of ["retrieval", "sts", "classification"]

        Returns:
            loss: Scalar loss
        """
        if task_type == "sts":
            # Symmetric loss for STS
            return self.infonce.forward_symmetric(
                query_embeds,
                pos_embeds,
                neg_embeds,
                neg_embeds,  # Use same negatives for both directions
            )
        else:
            # Asymmetric loss for retrieval and classification
            return self.infonce.forward(query_embeds, pos_embeds, neg_embeds)


def cosine_similarity_loss(
    embeds_a: torch.Tensor,
    embeds_b: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Simple cosine similarity loss for regression tasks (e.g., STS).

    Args:
        embeds_a: Embeddings for text A [batch_size, hidden_size]
        embeds_b: Embeddings for text B [batch_size, hidden_size]
        labels: Similarity scores [batch_size], range [0, 1] or [-1, 1]

    Returns:
        loss: MSE loss between predicted and true similarity
    """
    # Normalize
    embeds_a = F.normalize(embeds_a, p=2, dim=1)
    embeds_b = F.normalize(embeds_b, p=2, dim=1)

    # Compute cosine similarity
    cosine_sim = torch.sum(embeds_a * embeds_b, dim=1)

    # MSE loss
    loss = F.mse_loss(cosine_sim, labels)

    return loss


class TripletLoss(nn.Module):
    """
    Alternative: Triplet loss with hard negative mining.

    L = max(0, margin + sim(q, neg) - sim(q, pos))

    Args:
        margin: Margin parameter (default: 0.3)
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        query_embeds: torch.Tensor,
        pos_embeds: torch.Tensor,
        neg_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            query_embeds: [batch_size, hidden_size]
            pos_embeds: [batch_size, hidden_size]
            neg_embeds: [batch_size, num_negatives, hidden_size]

        Returns:
            loss: Scalar
        """
        # Normalize
        query_embeds = F.normalize(query_embeds, p=2, dim=1)
        pos_embeds = F.normalize(pos_embeds, p=2, dim=1)
        neg_embeds = F.normalize(neg_embeds, p=2, dim=2)

        # Positive similarity
        pos_sim = torch.sum(query_embeds * pos_embeds, dim=1)  # [batch_size]

        # Negative similarities
        query_expanded = query_embeds.unsqueeze(1)  # [batch_size, 1, hidden_size]
        neg_sim = torch.sum(query_expanded * neg_embeds, dim=2)  # [batch_size, num_negatives]

        # Take hardest negative (highest similarity)
        hardest_neg_sim, _ = torch.max(neg_sim, dim=1)  # [batch_size]

        # Triplet loss
        loss = F.relu(self.margin + hardest_neg_sim - pos_sim)

        return loss.mean()
