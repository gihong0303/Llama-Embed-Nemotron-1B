"""
Bi-directional Llama Model for Text Embedding

This module converts a causal (decoder-only) Llama model into a bi-directional encoder
by removing the causal attention mask, following the approach in Llama-Embed-Nemotron paper.
"""

import torch
import torch.nn as nn
from transformers import LlamaModel, LlamaConfig, LlamaPreTrainedModel
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)
from typing import Optional, Tuple, Union


class BiDirectionalLlamaAttention(LlamaAttention):
    """
    Llama attention with bi-directional (non-causal) masking.

    Key modification: Removes causal mask to allow tokens to attend to future positions.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with bi-directional attention (no causal mask).

        The key difference from standard Llama attention is that we do NOT apply
        a causal mask, allowing each token to attend to all other tokens in the sequence.
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings if available
        if hasattr(self, 'rotary_emb'):
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat k/v heads if necessary (for GQA)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)

        # Apply attention mask (but NOT causal mask)
        # Only apply padding mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Compute output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class BiDirectionalLlamaDecoderLayer(LlamaDecoderLayer):
    """
    Llama decoder layer with bi-directional attention.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Replace the attention module with bi-directional version
        self.self_attn = BiDirectionalLlamaAttention(config, layer_idx)


class BiDirectionalLlamaModel(LlamaPreTrainedModel):
    """
    Bi-directional Llama model for text embedding.

    This model removes the causal mask from Llama's attention mechanism,
    converting it from a decoder-only model to a bi-directional encoder.

    Architecture modifications from standard Llama:
    1. All attention layers use bi-directional (non-causal) masking
    2. All parameters are unfrozen for end-to-end fine-tuning
    3. Output: last layer hidden states with mean pooling

    Args:
        config: LlamaConfig

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        >>> model = BiDirectionalLlamaModel.from_pretrained("meta-llama/Llama-3.2-1B")
        >>>
        >>> inputs = tokenizer("Hello world", return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Replace all decoder layers with bi-directional versions
        self.layers = nn.ModuleList(
            [BiDirectionalLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, torch.FloatTensor]:
        """
        Forward pass through bi-directional Llama model.

        Returns:
            last_hidden_state: [batch_size, sequence_length, hidden_size]
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Create position ids
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Create attention mask for padding (but NOT causal mask)
        if attention_mask is not None:
            # Convert attention mask to proper format: [batch_size, 1, 1, seq_length]
            # This allows broadcasting across heads and query positions
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)
            # Inverted mask: 0 for valid positions, large negative for padding
            attention_mask = (1.0 - attention_mask) * torch.finfo(inputs_embeds.dtype).min

        hidden_states = inputs_embeds

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,  # past_key_value
                    output_attentions,
                    False,  # use_cache
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=output_attentions,
                    use_cache=False,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attns] if v is not None)

        # Return last_hidden_state
        return hidden_states

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a pre-trained Llama model and convert it to bi-directional.

        Args:
            pretrained_model_name_or_path: Path or model ID (e.g., "meta-llama/Llama-3.2-1B")

        Returns:
            BiDirectionalLlamaModel with loaded weights
        """
        # Load config
        config = LlamaConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Create bi-directional model
        model = cls(config)

        # Load weights from standard Llama model
        # The weights are compatible since we only changed the attention mask logic
        standard_llama = LlamaModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # Copy weights
        model.embed_tokens.load_state_dict(standard_llama.embed_tokens.state_dict())
        model.norm.load_state_dict(standard_llama.norm.state_dict())

        for i, layer in enumerate(model.layers):
            # Copy layer weights (attention and FFN weights are compatible)
            layer.load_state_dict(standard_llama.layers[i].state_dict(), strict=False)

        return model
