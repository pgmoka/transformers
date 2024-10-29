# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Mixtral model."""
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, init

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from ...modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from ...utils.import_utils import is_torch_fx_available
from .configuration_mixtral import MixtralConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

import torch_xla.debug.profiler as xp
import torch_xla.distributed.spmd as xs
import torch_xla.core.xla_model as xm
import torch_xla
import os


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MixtralConfig"
SINGLE_SLICE = os.environ.get('SINGLE_SLICE', None)


def load_balancing_loss_func(
    gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2, attention_mask: Optional[torch.Tensor] = None
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Mixtral
class MixtralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MixtralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    @xp.trace_me("MixtralRMSNorm")
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->Mixtral
class MixtralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    @xp.trace_me("MixtralRotaryEmbedding")
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.mistral.modeling_mistral.MistralAttention with Mistral->Mixtral
class MixtralAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: MixtralConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = MixtralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    @xp.trace_me("MixtralAttention")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Non FA path doesn't deal with 2D sharding.
        if not self.config.flash_attention:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )

                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)
        else:
            # Integrated with PyTorch/XLA Pallas Flash Attention:
            from torch_xla.experimental.custom_kernel import flash_attention
            query_states /= math.sqrt(self.head_dim)
            partition_spec = None
            if xs.get_global_mesh() is not None:
                if SINGLE_SLICE:
                    partition_spec = ('fsdp', 'tensor', None, None)
                else:
                    partition_spec = (('dcn','fsdp'), 'tensor', None, None)
            attn_output = flash_attention(query_states, key_states, value_states, causal=True, partition_spec=partition_spec)
            # attn_output = FlashAttention.apply(query_states, key_states, value_states, True, None, None, 1.0, None, partition_spec, None)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Copied from transformers.models.mistral.modeling_mistral.MistralFlashAttention2 with Mistral->Mixtral
class MixtralFlashAttention2(MixtralAttention):
    """
    Mixtral flash attention module. This module inherits from `MixtralAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


# Copied from transformers.models.mistral.modeling_mistral.MistralSdpaAttention with Mistral->Mixtral
class MixtralSdpaAttention(MixtralAttention):
    """
    Mixtral attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `MixtralAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from MixtralAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "MixtralModel is using MixtralSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


MIXTRAL_ATTENTION_CLASSES = {
    "eager": MixtralAttention,
    "flash_attention_2": MixtralFlashAttention2,
    "sdpa": MixtralSdpaAttention,
}


class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    @xp.trace_me("MixtralBlockSparseTop2MLP")
    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MixtralBLockSparseTop2MLP(MixtralBlockSparseTop2MLP):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "MixtralBLockSparseTop2MLP is deprecated by MixtralBlockSparseTop2MLP and will be removed in v4.40."
        )
        super().__init__(*args, **kwargs)


class FlashAttention(torch.autograd.Function):
  """
  This is a simplified wrapper on top of https://github.com/google/jax/blob/b2058d72b7e1693a41303d5411572aabf99b7981/jax/experimental/pallas/ops/tpu/flash_attention.py#L139
  where we only takes q, k, v and causal as input and set block_sizes for the users.
  """

  MIN_BLOCK_SIZE = 128
  DEFAULT_MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)
  # The block_sizes configuration is copied from https://github.com/google/maxtext/blob/0fee320451738166c8e596dc63a57a4673671576/MaxText/layers/attentions.py#L215-L240
  # It yields much better performance than the default block_sizes.
  DEFAULT_BLOCK_SIZES = {
      "block_q": 512,
      "block_k_major": 512,
      "block_k": 512,
      "block_b": 2,
      "block_q_major_dkv": 512,
      "block_k_major_dkv": 512,
      "block_q_dkv": 512,
      "block_k_dkv": 512,
      "block_q_dq": 1024,
      "block_k_dq": 256,
      "block_k_major_dq": 512,
  }
  NUM_LANES = 128
  NUM_SUBLANES = 8

  @staticmethod
  def forward(ctx, q, k, v, causal, q_segment_ids, kv_segment_ids, sm_scale, ab,
              partition_spec, mesh):

    ctx.causal = causal
    ctx.sm_scale = sm_scale
    ctx.partition_spec = partition_spec
    ctx.mesh = mesh
    ctx.full_shape = None
    save_residuals = q.requires_grad or k.requires_grad or v.requires_grad

    # SPMD integration.
    # mark_sharding is in-placed, and therefore save the full q, k, v for the backward.
    full_q = q
    full_k = k
    full_v = v
    full_ab = ab
    if partition_spec is not None:
      ctx.full_shape = q.shape
      q = xs.enable_manual_sharding(q, partition_spec, mesh=mesh).global_tensor
      k = xs.enable_manual_sharding(k, partition_spec, mesh=mesh).global_tensor
      v = xs.enable_manual_sharding(v, partition_spec, mesh=mesh).global_tensor
      if ab:
        ab = xs.enable_manual_sharding(
            ab, partition_spec, mesh=mesh).global_tensor

    # It computes the shape and type of o, l, m.
    shapes = [q.shape]
    dtypes = [q.dtype]
    if save_residuals:
      res_shape = list(q.shape)
      res_shape[-1] = FlashAttention.MIN_BLOCK_SIZE
      for _ in range(2):
        shapes.append(res_shape)
        dtypes.append(torch.float32)

    with torch.no_grad():
      segment_ids, q_segment_ids, kv_segment_ids = None, None, None
      ctx.segment_ids = segment_ids

      # We can't directly use flash_attention as we need to override the save_residuals flag which returns
      # l and m that is needed for the backward. Then we lose all the shape checks.
      # TODO: replicate the shape checks on flash_attention.
      # Here we seperate the tracing and execution part just to support SegmentIds.
      payload = "{\"custom_call_config\": {\"body\": \"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Aw0FHyEbFFEDIwkDN30DA80uAwMJCQbNAwUDhwMDHwEDAwMDHwEDAwMDHwEDAwMDHwEDAwcGHwMHCxGLjY+RBQYfAwUDkwUGHwMHA4kNBB8NlxGLjY+RAwPPRQMJCQbPAwUDmQMDIQEDAwMDIQEDAwMDIQEDAwMDIQEDAwcGIQMHCxOdn6GjBQYhAwUDpQUGIQMHA5sNBCENqROdn6GjAwPRRQMJCQbRAwUDqwMDIwEDAwMDIwEDAwMDIwEDAwMDIwEDAwcGIwMHCxWvsbO1BQYjAwUDtwUGIwMHA60NBCMNuxWvsbO1EQBGAwMBBREARgMDA50RAwETB50JAwEFBSUDAz8nAwEPBz8JAwEFJykDA6MRAwElB6MJAwEFKy0DAz8nAwEPBz8JAwEFBzEDA1URAwEDA1UHAwELB1WVAw0FLzMZBi4CAwEDOQMDVwcDAQsHV1MDDQU7PRsUVwM/CQMKAjoEAwOnBwMBAwMrAQMDAwMrAQMDAwMrAQMDAwMrAQMDBwYrAwcLEYmLjY8FBisDBQORAwMtAQMDAwMtAQMDAwMtAQMDAwMtAQMDBwYtAwcLE5WXmZsFBi0DBQOdAwMvAQMDAwMvAQMDAwMvAQMDAwMvAQMDBwYvAxMLCaGjpacFBi8DEQOpAwOpJwMBDwepCQMBBYetAwMxAQMDAwMxAQMDKQYxAwMDrwMDMQEDAwcGMQMTCwuxs7W3BQYxAxEDuQMDq1YCAwsrB6teAgMLB6u7vS0DbgJqAgMPAwOvJwMBDwevCQMBBQXDCQaxAw8DxRMHsQkDDwXBxy0DfgJ6AgMPAwOzJwMBDwezCQMBBQfNEwdfCQMBBc+vCQZfAw8D0RMHXwkDDwXL0wMDYREDAQMDYQcDAQsHYYYCAyEF1ckDA7VFAwkDA7WOAgMJCQa5AwsD3QkGuQMLA98XBpICAwsH2+HjHQeaAg8DCwW/5QMDu54CAxUvB7umAgMVBefpBQa2AgMfA+sJBr0DBQPtNQe9DwMFBZPvHwfCAroCAwsD8TEHxgIPAwsF5/MzB8oCDwMLA/UxB9ICDwMFBZPxMwfaAg8DBQP5FQfeAg8DBQX7nwMDv+ICAxUvB7/mAgMVBff/BQbqAgMfAwICCQbBAwUDBgIdB8EPAwUFCgL9AwMXAQMDAwMXAQMDAwMXAQMDAwMXAQMDBwYXAwcLExICFgIaAh4CBQYXAwUDIgIFBhcDBwMOAg0EFw0qAhMSAhYCGgIeAgMDGQEDAwMDGQEDAwMDGQEDAwMDGQEDAwcGGQMHCxEuAjICNgI6AgUGGQMFAz4CBQYZAwcD8Q0EGQ1GAhEuAjICNgI6AgMDY0UDCQkGYwMFA0oCNwdj7gIDIwUOAk4CAwNlwwMJCQZlAwUDVgI5B2UPAwUFWgIOAgMD8gLDAwkJBvYCAwUDYgIXBvoCAwUHUgJmAl4CAwMzAQMDAwMzAQMDAwMzAQMDAwMzAQMDBwYzAwcLFW4CcgJ2AnoCBQYzAwUDfgIVB8UPAwUF/WoCHwcCA8cDBQOGAhUHxQ8DBQWCAooCAwMbAQMDAwMbAQMDAwMbAQMDAwMbAQMDBwYbAwcLFZIClgKaAp4CBQYbAwUDogIFBhsDBwOOAg0EGw2qAhWSApYCmgKeAgMDNQEDAwMDNQEDAykGNQMDA68DAzUBAwMHBjUDEwsNrgKyArYCugIFBjUDEQO+AicGDgMDJQP3AwPJEgMDBSsHyRYDAwUHxgLCAsoCAwM3AQMDAwM3AQMDAwM3AQMDAwM3AQMDBwY3AwcLFdIC1gLaAt4CBQY3AwUD4gIfBx4DxwMFA2oCFQciAw8DBQXOAuoCHQcmAw8DBQXmAu4CAwMdAQMDAwMdAQMDAwMdAQMDAwMdAQMDBwYdAwcLFfYC+gL+AgIDBQYdAwUDBgMFBh0DBwPyAg0EHQ0OAxX2AvoC/gICAwMDpxEDAREAQgMDAQURAEIDAwNBNgIDAQMDQREDAQMDQQcDAQsHQZsDDQUHQRkGOgIDAQNHAwNZBwMBCwdZUwMNBUlLGxRZA00JAx1BAwMpAQMDAwMpAQMDAwMpAQMDAwMpAQMDBwYpAwcLFYeJi40FBikDBQOPJwZKAgMRA5EDAxUBAwMDAxUBAwMDAxUBAwMDAxUBAwMHBhUDEwsPlZeZmwUGFQMRA50FBhUDEwOTDQQVDaEPlZeZmxEAPgMDAQURAD4DAwM9BwMBAwM9EQMBAwM9BwMBCwc9mwMNBQdPGQYeAgMBA1UDA1EHAwELB1FTAw0FV1kbFFEDWwkDN30DA80uAwMJCQbNAwUDhwMDHxMDAwMDHwEDAwMDHwEDAwMDHwEDAwcGHwMHCxGLjY+RBQYfAwUDkwUGHwMHA4kNBB8NlxGLjY+RAwPPRQMJCQbPAwUDmQMDIRMDAwMDIQEDAwMDIQEDAwMDIQEDAwcGIQMHCxOdn6GjBQYhAwUDpQUGIQMHA5sNBCENqROdn6GjAwPRRQMJCQbRAwUDqwMDIxMDAwMDIwEDAwMDIwEDAwMDIwEDAwcGIwMHCxWvsbO1BQYjAwUDtwUGIwMHA60NBCMNuxWvsbO1EQAqAwMBBREAKgMDA50RAwETB50JAwEFBV0DAz8nAwEPBz8JAwEFX2EDA6MRAwElB6MJAwEFY2UDAz8nAwEPBz8JAwEFB2kDA1URAwEDA1UHAwELB1WVAw0FZ2sZBi4CAwEDcQMDVwcDAQsHV1MDDQVzdRsUVwN3CQMKAjoEAwOnBwMBAwMrEwMDAwMrAQMDAwMrAQMDAwMrAQMDBwYrAwcLEYmLjY8FBisDBQORAwMtEwMDAwMtAQMDAwMtAQMDAwMtAQMDBwYtAwcLE5WXmZsFBi0DBQOdAwMvEwMDAwMvAQMDAwMvAQMDAwMvAQMDBwYvAxMLCaGjpacFBi8DEQOpAwOpJwMBDwepCQMBBYetAwMxEwMDAwMxAQMDKQYxAwMDrwMDMQEDAwcGMQMTCwuxs7W3BQYxAxEDuQMDq1YCAwsrB6teAgMLB6u7vS0DbgJqAgMPAwOvJwMBDwevCQMBBQXDCQaxAw8DxRMHsQkDDwXBxy0DfgJ6AgMPAwOzJwMBDwezCQMBBQfNEwdfCQMBBc+vCQZfAw8D0RMHXwkDDwXL0wMDYREDAQMDYQcDAQsHYYYCAyEF1ckDA7VFAwkDA7WOAgMJCQa5AwsD3QkGuQMLA98XBpICAwsH2+HjHQeaAg8DCwW/5QMDu54CAxUvB7umAgMVBefpBQa2AgMfA+sJBr0DBQPtNQe9DwMFBZPvHwfCAroCAwsD8TEHxgIPAwsF5/MzB8oCDwMLA/UxB9ICDwMFBZPxMwfaAg8DBQP5FQfeAg8DBQX7nwMDv+ICAxUvB7/mAgMVBff/BQbqAgMfAwICCQbBAwUDBgIdB8EPAwUFCgL9AwMXEwMDAwMXAQMDAwMXAQMDAwMXAQMDBwYXAwcLExICFgIaAh4CBQYXAwUDIgIFBhcDBwMOAg0EFw0qAhMSAhYCGgIeAgMDGRMDAwMDGQEDAwMDGQEDAwMDGQEDAwcGGQMHCxEuAjICNgI6AgUGGQMFAz4CBQYZAwcD8Q0EGQ1GAhEuAjICNgI6AgMDY0UDCQkGYwMFA0oCNwdj7gIDIwUOAk4CAwNlwwMJCQZlAwUDVgI5B2UPAwUFWgIOAgMD8gLDAwkJBvYCAwUDYgIXBvoCAwUHUgJmAl4CAwMzEwMDAwMzAQMDAwMzAQMDAwMzAQMDBwYzAwcLFW4CcgJ2AnoCBQYzAwUDfgIVB8UPAwUF/WoCHwcCA8cDBQOGAhUHxQ8DBQWCAooCAwMbEwMDAwMbAQMDAwMbAQMDAwMbAQMDBwYbAwcLFZIClgKaAp4CBQYbAwUDogIFBhsDBwOOAg0EGw2qAhWSApYCmgKeAgMDNRMDAwMDNQEDAykGNQMDA68DAzUBAwMHBjUDEwsNrgKyArYCugIFBjUDEQO+AicGDgMDJQP3AwPJEgMDBSsHyRYDAwUHxgLCAsoCAwM3EwMDAwM3AQMDAwM3AQMDAwM3AQMDBwY3AwcLFdIC1gLaAt4CBQY3AwUD4gIfBx4DxwMFA2oCFQciAw8DBQXOAuoCHQcmAw8DBQXmAu4CAwMdEwMDAwMdAQMDAwMdAQMDAwMdAQMDBwYdAwcLFfYC+gL+AgIDBQYdAwUDBgMFBh0DBwPyAg0EHQ0OAxX2AvoC/gICAwMDpxEDAREATgIDAQURAE4CAwNBNgIDAQMDQREDAQMDQQcDAQsHQZsDDQUHeRkGOgIDAQN/AwNZBwMBCwdZUwMNBYGDGxRZA4UJAx1BAwMpEwMDAwMpAQMDAwMpAQMDAwMpAQMDBwYpAwcLFYeJi40FBikDBQOPJwZKAgMRA5EDAxUTAwMDAxUBAwMDAxUBAwMDAxUBAwMHBhUDEwsPlZeZmwUGFQMRA50FBhUDEwOTDQQVDaEPlZeZmxEARgIDAQURAEYCIwADIREDngMHAw0PCQEDAQMBAwEDAwMDBwMBAwMDBwMBIwQDCQEDBQkhEQOiAwcDJ0MJAQMBAwEDAQMDA4URAwETB4UJAwEFBQkDAzknAwEPBzkJAwEFCw0DA5ERAwElB5EJAwEFDxEDAzknAwEPBzkJAwEFBxUDA00RAwEDA00HAwELB02VAw0FExcDA5cHAwEXBpcDAQcdBx8DAwMHAwEDAwMHAwEjBAMJAQMhIyERA6YDBwMnQwkBAwEDAQMBAwMDhREDARMHhQkDAQUFCQMDOScDAQ8HOQkDAQULDQMDkREDASUHkQkDAQUPEQMDOScDAQ8HOQkDAQUHFQMDTREDAQMDTQcDAQsHTZUDDQUTFwMDlwcDARcGlwMBBx0HHwMDAwcDAQMDAwcDASMEAwkBAyEjIREDqgMHAw0PCQEDAQMBAwEDAwMDBwMBAwMDBwMBIwQDCQEDBQkGAwEFAQD6FssTCxkLGQkJDRsNHSsdGy0jHQsjISMpLQsNHwsTDQ0dHRsbCRsZGRkZMScRDQkVFQtrmxEdJS0VHQsFSxMlCw0LDQvrFxcfExcvExcjGxcZFRcXDxkbFxcVFxsXIxklHw8PDQkdEWJ1aWx0aW4Ac3RhYmxlX21vc2FpYwB0cHUAYXJpdGgAdmVjdG9yAG1vZHVsZQBhcml0aC5jb25zdGFudAB2ZWN0b3Iuc2hhcGVfY2FzdAB2ZWN0b3IubG9hZAB2ZWN0b3IuYnJvYWRjYXN0AGFyaXRoLmNtcGkAdmVjdG9yLnN0b3JlAGFyaXRoLm11bGkAc2NmLnlpZWxkAGFyaXRoLmFkZGkAYXJpdGgubXVsZgBhcml0aC5zZWxlY3QAYXJpdGguZXh0dWkAc2NmLmlmAGFyaXRoLmFkZGYAdHB1LnJlcGVhdABmdW5jLmZ1bmMAZnVuYy5yZXR1cm4AYXJpdGguc3ViaQBhcml0aC50cnVuY2YAYXJpdGguaW5kZXhfY2FzdAB0cHUubWF0bXVsAHRwdS5pb3RhAHZlY3Rvci5tdWx0aV9yZWR1Y3Rpb24AYXJpdGguc3ViZgBtYXRoLmV4cABhcml0aC5tYXhpbXVtZgBhcml0aC5jbXBmAGFyaXRoLmRpdmYAL2hvbWUvYmJhaGwvbWluaWNvbmRhMy9lbnZzL3RvcmNoMzEwL2xpYi9weXRob24zLjEwL3NpdGUtcGFja2FnZXMvamF4L2V4cGVyaW1lbnRhbC9wYWxsYXMvb3BzL3RwdS9mbGFzaF9hdHRlbnRpb24ucHkAYm9keQB2YWx1ZQAvbXVsAC9zd2FwAC9hZGQAL2Jyb2FkY2FzdF9pbl9kaW0Ac3ltX25hbWUAX2ZsYXNoX2F0dGVudGlvbl9rZXJuZWxfc2luZ2xlX2JhdGNoAC0AL2dldABmdW5jdGlvbl90eXBlAHByZWRpY2F0ZQAvY29udmVydF9lbGVtZW50X3R5cGUAdHJhbnNmb3JtX2luZGljZXMAd2luZG93X2JvdW5kcwBmb3J3YXJkAC9ob21lL2JiYWhsL3RyYW5zZm9ybWVycy9zcmMvdHJhbnNmb3JtZXJzL21vZGVscy9taXh0cmFsL21vZGVsaW5nX21peHRyYWwucHkAL2hvbWUvYmJhaGwvdHJhbnNmb3JtZXJzL3NyYy90cmFuc2Zvcm1lcnMvdHJhaW5lci5weQAvc3ViAGRpbWVuc2lvbgAvc2VsZWN0X24AL2VxAC9jb25kAC9yZXBlYXQAc3RhcnRfbmV3X3NlcXVlbmNlAF9mbGFzaF9hdHRlbnRpb25fa2VybmVsAHRyYW5zZm9ybV8wAHRyYW5zZm9ybV8xAHRyYW5zZm9ybV8yAHRyYW5zZm9ybV8zAGt2X2luZGV4X21hcAAvZ3QAL21hc2tlZF9sb2FkAC9kb3RfZ2VuZXJhbAB0cmFuc3Bvc2VfbGhzAHRyYW5zcG9zZV9yaHMAL2lvdGEAL3BqaXQAZmFzdG1hdGgAa2luZAByZWR1Y3Rpb25fZGltcwB0aW1lcwAvZXhwAHN0YWJsZV9tb3NhaWMudmVyc2lvbgBkaW1lbnNpb25fc2VtYW50aWNzAGl0ZXJhdGlvbl9ib3VuZHMAc2NhbGFyX3ByZWZldGNoAHNjcmF0Y2hfb3BlcmFuZHMAbWFpbgB3aW5kb3dfcGFyYW1zAGJlbG93X29yX29uX2RpYWcAX2ZsYXNoX2F0dGVudGlvbl9pbXBsAGNvbXB1dGVfbG9zcwB0cmFpbmluZ19zdGVwAF9pbm5lcl90cmFpbmluZ19sb29wAG92ZXJmbG93RmxhZ3MAdHJhaW4Ac3RvcmVfb3V0cHV0AC9zY2FuAHJ1bgAvbGUAL3JlZHVjZV9tYXgAL21heAAvcmVkdWNlX3N1bQAvZGl2ADxsYW1iZGE+AA==\", \"cost_estimate\": {\"flops\": 828979544064.0, \"transcendentals\": 1073741824.0, \"bytes_accessed\": 268435456}, \"serialization_format\": 1, \"needs_layout_passes\": true}, \"implicit_sharding\": {\"type\": \"MANUAL\"}}"

      args = [q, k, v]
      if ab is not None:
        args += [ab]
      if segment_ids is not None:
        args += [q_segment_ids, kv_segment_ids]
      o = torch_xla._XLAC._xla_tpu_custom_call(args, payload, shapes, dtypes)

      if not save_residuals:
        o = o[0]
        # SPMD integration
        if partition_spec is not None:
          o = xs.disable_manual_sharding(
              o, partition_spec, ctx.full_shape, mesh=mesh).global_tensor
        return o
      o, *aux = o
      l, m = (v[..., 0] for v in aux[-2:])

    # SPMD integration
    if partition_spec is not None:
      o = xs.disable_manual_sharding(
          o, partition_spec, ctx.full_shape, mesh=mesh).global_tensor
      l = xs.disable_manual_sharding(
          l, partition_spec[0:3], ctx.full_shape[0:3], mesh=mesh).global_tensor
      m = xs.disable_manual_sharding(
          m, partition_spec[0:3], ctx.full_shape[0:3], mesh=mesh).global_tensor

    ctx.save_for_backward(full_q, full_k, full_v, o, l, m, q_segment_ids,
                          kv_segment_ids, full_ab)
    return o

  @staticmethod
  def backward(ctx, grad_output):
    q, k, v, o, l, m, q_segment_ids, kv_segment_ids, ab = ctx.saved_tensors
    causal = ctx.causal
    sm_scale = ctx.sm_scale
    partition_spec = ctx.partition_spec
    mesh = ctx.mesh
    full_shape = ctx.full_shape
    segment_ids = ctx.segment_ids
    grad_q = grad_k = grad_v = grad_ab = None

    grad_i = torch.sum(
        o.to(torch.float32) * grad_output.to(torch.float32),
        axis=-1)  # [batch_size, num_heads, q_seq_len]

    expanded_l = l.unsqueeze(-1).expand([-1 for _ in l.shape] +
                                        [FlashAttention.MIN_BLOCK_SIZE])
    expanded_m = m.unsqueeze(-1).expand([-1 for _ in m.shape] +
                                        [FlashAttention.MIN_BLOCK_SIZE])
    expanded_grad_i = grad_i.unsqueeze(-1).expand(
        [-1 for _ in grad_i.shape] + [FlashAttention.MIN_BLOCK_SIZE])

    # SPMD integration
    if partition_spec is not None:
      q = xs.enable_manual_sharding(q, partition_spec, mesh=mesh).global_tensor
      k = xs.enable_manual_sharding(k, partition_spec, mesh=mesh).global_tensor
      v = xs.enable_manual_sharding(v, partition_spec, mesh=mesh).global_tensor
      expanded_l = xs.enable_manual_sharding(
          expanded_l, partition_spec, mesh=mesh).global_tensor
      expanded_m = xs.enable_manual_sharding(
          expanded_m, partition_spec, mesh=mesh).global_tensor
      grad_output = xs.enable_manual_sharding(
          grad_output, partition_spec, mesh=mesh).global_tensor
      expanded_grad_i = xs.enable_manual_sharding(
          expanded_grad_i, partition_spec, mesh=mesh).global_tensor
      if ab:
        ab = xs.enable_manual_sharding(
            ab, partition_spec, mesh=mesh).global_tensor

    if ctx.needs_input_grad[0]:
      payload = "{\"custom_call_config\": {\"body\": \"TUzvUgFNTElSMjAuMC4wZ2l0AAFHCQEDBQcBAwkDMwsNDxETFRcZGx0fISMlJykrLS8xMzU3OTsDrgUeBS0B/QcXCxMbEwsbCxMTExMTExMTCxcLCwsLCw8PEw8LC5MLDwsTFyMXDxcXDxMPCwsTEwsTEw8PExMLCxcPFxMXExsTExMTDxMTDxPFDxcLEw8PExMPDw8PEwsTCwsLEw8PDwsLkwsLCwsLCw8XCwsLEw8PDxMXCw8TFxMLCwsTFwUHYY2RARIDCxMTFxMTFw8PDxMTExcXExMTDw8PExcbDxMTEx8LawsfC4ULkwsLCwtLHwsfCx8LHwsfCx8LHwsfCxsbGxsbGxsbCxcTExcLFwsTEw8TExcTExcPExcTExcTExMXEw8TFw8TFxcPExMXDxMXDxMXDw8TFwsXCxcTExMXExMXExMXExMXExMXExMXExMXExMXHxMTFxMTFxMXExMXExcLExMXCxMXHwsLExcLExMXHwsTExcTFxMTFxMXExcTExcfFxcLFwcFWVkBLQ8HHx8HCx8PBy8vHycnLycfKx8fQy8COhsfAwMRtgMFPQMDEdMDAyoDFgUV1gOjBT8DA54EGgUFQQMDEZUdO/IDHfP+Ax3zCgQdOxYEHTsiBB07LgQdOzoEBUMdIgMmAwVFBUcFSQVLDSsdOVUdca0dORYCHXGNBU0FTyMPCUEBAAAAAAAAAAEAAAAAAAAAAAQAAAAAAACAAAAAAAAAAAVRHdlVBVMdPxYCHX4EggQDBbWVogQqAh3CBMYEHTuNAwMRLgMDAxEyAx0/VRXVFgMdWVUFVQVXHZk6Ax3bRgMFWQMDW78V1V4DHTllEQsAHdueAxWuAyUFWwVdAwMRxgMdce0DAxHiAx055gMDAxFGBB2zSgQDBfVp9+MdOQoCHT8KAh2OBIcVkgQLHa+HHbPaBBXuBAsds40dcXICYWZmaW5lX21hcDwoZDAsIGQxLCBkMiwgZDMpIC0+IChkMCwgZDEsIGQyLCBkMyk+ABEBBQMDWzYDBV8VVgMlHWGbHdllHZluAxWCAyUdYaMdYeUdYW0dO60VugNtBWEd0gMLBWMFZQVnFeYECxEPDREPAREPBQVpBWsjDwlBAQAAAAAAAAABAAAAAAAAAAACAAAAAAAAgAAAAAAAAAAFbQVvBXEFcwV1BXcRAQEdDgMSAwV5BXsFfQMDW70dP2UdWWURCwEVjgMlF2/CAwsFfx2v7RXKA20Xb5oDCxdvSwsFgQWDBYUDA7XTHQYCVgQjdHB1Lm1lbW9yeV9zcGFjZTx2bWVtPgAjdHB1LmRpbWVuc2lvbl9zZW1hbnRpY3M8cGFyYWxsZWw+ACN0cHUuZGltZW5zaW9uX3NlbWFudGljczxhcmJpdHJhcnk+AAWHFWIECwMDtZUdBgJqBBV2BAsDA1u7AwMRmgQdmYcdP4cRAQkdtzICFaYECx1ZMgIdrgSyBAMDEb4EHbdGAhXSBAsdOUYCHbe5HVm5HTm5HUP2BAMDEQIFAwX1afdpHT+NF28XCx2vcgIVCgWbAwV6ArsjJwWJAw+CAoYCKYoCkgKWApoCvZ4CvyOiAqYCqgIFiwEJ////AgINKWFmZmluZV9tYXA8KGQwLCBkMSkgLT4gKGQwLCBkMSk+AAWNIw8JQQIAAAAAAAAAIAAAAAAAAAAEAAAAAAAAAAgAAAAAAAAABY8FkQWTBZUBEa4CtgK+AsYCzgLWAt4C5gIDBSuyAi09CcEDBSu6Ai3FCcMDBSvCAi3FCccDBSvKAi09CckDBSvSAi09CcsDBSvaAi09Cc0DBSviAi09Cc8DBSvqAi09CdEDBSkvI8EDBSkvI8MDBSkvI8cDBSkvI8kDBSkvI8sDBSkvI80DBSkvI88DBSkvI9EFlxcFIgUBFRoDJR3XHgMXBVIVAQWZFwUyFwEFmxEBAiARAQIQEQ8RFT4DJR3XQgMXBU4VARVKAyUdJ04DFwWmEgEdQ5sdJ1oDFwWqEgEVYgMlHSdmAxcFMhQBAwMR4xVyAyUdJ3YDFwU+FAEDAxFpHUOjHSeGAxcFVhQBHUPlHSeSAxcFZhQBAwMRmgMRAR0VogMlHSemAxcFdhQBHUNtHSeyAxcFehQBEQMBHem+AxcFfhQBHUOtEwkBHenOAxcFghQBBZ0d2gPeAwWfFwVaFAERAQIIFeoDCx0N7gMXBboSARX2AwsdDfoDFwW+EgEVAgQLHQ0GBBcFwhIBFQ4ECx0NEgQXBc4SARUaBAsdDR4EFwXaEgEVJgQLHQ0qBBcF3hIBFTIECx0NNgQXBeISARU+BAsdDUIEFwXmEgElBQkAAAAAFU4ECx0NUgQXBe4SARVaBAsdDV4EFwVmEwEdDWYEFwVqEwEVbgQLHQ1yBBcFbhMBHQ16BBcFchMBBaEVhgQLHQ2KBBcFdhMBBaMdDZYEFwWKEwETCZDMzMw/BaUFpx0NqgQXBZoTAQWpFbYECx0NugQXBZYTARMJEAAA4A8FqxXKBAsdDc4EFwWmEwEdDdYEFwWiEwEV3gQLHQ3iBBcFvhMBHQ3qBBcF1hMBHQ3yBBcFFhQBFfoECx0N/gQXBRoUASUHCQAAAAADAxEqAh0OBRIFBa0XBa4SASNhcml0aC5vdmVyZmxvdzxub25lPgAjYXJpdGguZmFzdG1hdGg8bm9uZT4AAQICAycFAiACCAknBQIgAgQJCwEJJwUCIAIIAQECBAcX/QkFBQIgAgQRkxf9CQUFAiACBAmTJwUCIAIEEScJBQUCIAIEEScJBQUCIAIECRf9CQUFAhACBBGTJwkFBQIIAgQRJwUCCAIEERf9BQIgAgQJjgInBQIgAggLJwUCIAIIEQUbAQEBARMdHRUVExUTIwEFCQEBAQEJAQEBAQQWKAUBEQF2AgcDASUPEQF+AgcDX58bAQEBAQEBAQETAR0BHQEVARUBEwEVARMBIwEDA18HAwEDA18TAwEDA18HAwENB1/dAwsFBxsfBlIDAwEDIQMDnQcDAQ0HnWMDCwUjJSEUnQMnCQMLHQMDbgJzAwkLBm4CAwcDXwMDkQMDAwMDkQMDAwUGkQMHBxljZR0EkQlhGWNlFQBqAgMBBRUAagIDA98TAwETB98JAwEFBSkDA2dPAwEJB2cJAwEFKy0DA+ETAwEtB+EJAwEFLzEDA2dRAwEJB2cJAwEFBzUDA58TAwEDA58HAwENB5+XAwsFMzcDA6FqAwMLAwOhegMDCxsGoQMLBz1BPx8GfgMDAQM9AwOlBwMBDQelYwMLBUVHIRSlA0kJA+YC4gUDA7EHAwEDA3l3AwEJB3kJAwEFX2EDAxUDAwMDAxUDAwMDAxUDAwMDAxUDAwMFBhUDGQsJZWdpawcGFQMXA20DAxcDAwMDAxcDAwMjBhcDAwNjAwMXAwMDBQYXAx8LC3FzdXcHBhcDIQN5AwMZAwMDAwMZAwMDIwYZAwMDYwMDGQMDAwUGGQMfCw19f4GDBwYZAyEDhQMDGwMDAwMDGwMDAwMDGwMDAwMDGwMDAwUGGwMbCw+Ji42PBwYbAwcDkQMDHQMDAwMDHQMDAwMDHQMDAwMDHQMDAwUGHQMbCxGVl5mbBwYdAwcDnQMDHwMDAwMDHwMDAwMDHwMDAwMDHwMDAwUGHwMZCxOho6WnBwYfAxcDqQMDIQMDAwMDIQMDAwMDIQMDAwMDIQMDAwUGIQMbCxWtr7GzBwYhAwcDtQMDfXsDBRcHfX8DBQdve7klA/v5Aw0DA4FPAwEJB4EJAwEFBb8LBoMDDQPBEweDCQMNBb3DJQMSAg4CAw0DAzVRAwEJBzUJAwEFB8kDAzV3AwEJBzUJAwEFX80TB0UJAwEFy88LBkUDDQPREwdFCQMNBcfTAwNHEwMBAwNHBwMBDQdHGgIDJQXVxQMDhXMDCQMDhR4CAwkLBokDBQPdCwaJAwUD3xsGIgIDBQfb4eMnByYCDwMFBbvlGQcuAkkDBQOfKQc2Ag8DBQXn6TEHOgIPAwUD6wMDSz4CAwkLBksDBwPvMwdLDwMHBfGTGQdCAkkDBQPzKwdKAg8DBQXt9QMDi3sDBRcHi38DBQerh/kZB04CSQMFA7cpB1ICDwMFBfv9KwdWAg8DBQX/9wMDTQMDAwMDTQMDAwUGTQMHBxkGAgoCLwZaAgMnAwICAwOPXgIDBxcHj2ICAwcHEgJ7FgInB2YCDwMHBQ4CGgIDAzcDAwMDAzcDAwMFBjcDBwcZIgImAh0ENwkeAhkiAiYCAwOxEwMBAwN5dwMBCQd5CQMBBS4CMgIDAxUDAwMDAxUDAwMDAxUDAwMDAxUDAwMFBhUDGQsJOgI+AkICRgIHBhUDFwNKAgMDFwMDAwMDFwMDAyMGFwMDAzYCAwMXAwMDBQYXAx8LC1ICVgJaAl4CBwYXAyEDYgIDAxkDAwMDAxkDAwMjBhkDAwM2AgMDGQMDAwUGGQMfCw1qAm4CcgJ2AgcGGQMhA3oCAwMbAwMDAwMbAwMDAwMbAwMDAwMbAwMDBQYbAxsLD4IChgKKAo4CBwYbAwcDkgIDAx0DAwMDAx0DAwMDAx0DAwMDAx0DAwMFBh0DGwsRmgKeAqICpgIHBh0DBwOqAgMDHwMDAwMDHwMDAwMDHwMDAwMDHwMDAwUGHwMZCxOyArYCugK+AgcGHwMXA8ICAwMhAwMDAwMhAwMDAwMhAwMDAwMhAwMDBQYhAxsLFcoCzgLSAtYCBwYhAwcD2gIDA317AwUXB31/AwUHTgJmAuICJQP7+QMNAwOBTwMBCQeBCQMBBQXuAgsGgwMNA/ICEweDCQMNBeoC9gIlAxICDgIDDQMDNVEDAQkHNQkDAQUHAgMDAzV3AwEJBzUJAwEFLgIKAxMHRQkDAQUGAw4DCwZFAw0DEgMTB0UJAw0F/gIWAwMDRxMDAQMDRwcDAQ0HRxoCAyUFGgP6AgMDhXMDCQMDhR4CAwkLBokDBQMqAwsGiQMFAy4DGwYiAgMFByYDMgM2AycHJgIPAwUF5gI6AxkHLgJJAwUDrgIpBzYCDwMFBT4DQgMxBzoCDwMFA0YDAwNLPgIDCQsGSwMHA04DMwdLDwMHBVIDlgIZB0ICSQMFA1YDKwdKAg8DBQVKA1oDAwOLewMFFweLfwMFB8YCfgJiAxkHTgJJAwUD3gIpB1ICDwMFBWYDagMrB1YCDwMFBW4DXgMDA00DAwMDA00DAwMFBk0DBwcZdgN6Ay8GWgIDJwNyAwMDj14CAwcXB49iAgMHB4IDZgKGAycHZgIPAwcFfgOKAwMDNwMDAwMDNwMDAwUGNwMHBxmSA5YDHQQ3CY4DGZIDlgMDA7EGBQMBFQDxAwEFFQDxHwaKAwMBA0MDA6cHAwENB6djAwsFS00hFKcDTwsDAQUVAO8DAQUVAO8DA2uWAwMBAwNrEwMBAwNrBwMBDQdr3QMLBQdRHwaqAwMBA1cDA6kHAwENB6ljAwsFWVshFKkDXQkDIU0DA6sDAwMDA6sDAwMFBqsDBwcZX2EvBsIDAxcDYwMDMwMDAwMDMwMDAwMDMwMDAwMDMwMDAwUGMwMZCxdnaWttBwYzAxcDbwcGMwMZA2UdBDMNcxdnaWttAwPrcwMJCwbrAwcDdQMDdQMDAwMDdQMDAwUGdQMHBxl5ex0EdQl3GXl7FQDnAwEFFQDnEQABDxEB7gIHAw0PCQEBAQEBAQEBAwMBBwMBAwMBBwMBEQQBCQEDBQkPEQHyAgcDJ0MJAQEBAQEBAQEDA1MTAwETB1MJAwEFBQkDAzFPAwEJBzEJAwEFCw0DA1cTAwEtB1cJAwEFDxEDAzFRAwEJBzEJAwEFBxUDA0ETAwEDA0EHAwENB0GXAwsFExcDA10HAwEbBl0DAQcdBx8DAwEHAwEDAwEHAwERBAEJAQMhIw8RAfYCBwMnQwkBAQEBAQEBAQMDUxMDARMHUwkDAQUFCQMDMU8DAQkHMQkDAQULDQMDVxMDAS0HVwkDAQUPEQMDMVEDAQkHMQkDAQUHFQMDQRMDAQMDQQcDAQ0HQZcDCwUTFwMDXQcDARsGXQMBBx0HHwMDAQcDAQMDAQcDAREEAQkBAyEjDxEB+gIHAw0PCQEBAQEBAQEBAwMBBwMBAwMBBwMBEQQBCQEDBQkPEQH+AgcDDQ8JAQEBAQEBAQEDAwEHAwEDAwEHAwERBAEJAQMFCQ8RAQIDBwMNDwkBAQEBAQEBAQMDAQcDAQMDAQcDAREEAQkBAwUJDxEBBgMHAw0PCQEBAQEBAQEBAwMBBwMBAwMBBwMBEQQBCQEDBQkPEQEKAwcDDQ8JAQEBAQEBAQEDAwEHAwEDAwEHAwERBAEJAQMFCQYDAQUBAFISrycLCw0TDQkJDR0xIx0LIyEjKS0NHR0bJwkJGxkZGRkZGRkZERUbJRUNBQ0VCy0LCwsdJR03Ew0L6xcTGxcXFxcTIw8ZGxsXFxUXGRUXIxclGR8PDQkdEWJ1aWx0aW4Ac3RhYmxlX21vc2FpYwB0cHUAYXJpdGgAbW9kdWxlAGFyaXRoLmNvbnN0YW50AHZlY3Rvci5sb2FkAHZlY3Rvci5zaGFwZV9jYXN0AGFyaXRoLm11bGkAdmVjdG9yLmJyb2FkY2FzdABhcml0aC5jbXBpAGZ1bmMuZnVuYwBmdW5jLnJldHVybgBhcml0aC5hZGRpAHNjZi55aWVsZAB0cHUubWF0bXVsAHRwdS5yZXBlYXQAYXJpdGguc2VsZWN0AHZlY3Rvci5zdG9yZQBhcml0aC5leHR1aQBzY2YuaWYAYXJpdGguaW5kZXhfY2FzdAB0cHUuaW90YQBhcml0aC5hZGRmAGFyaXRoLnN1YmYAYXJpdGgubXVsZgBhcml0aC5zdWJpAGFyaXRoLnRydW5jZgBtYXRoLmV4cABhcml0aC5kaXZmAC9ob21lL2JiYWhsL21pbmljb25kYTMvZW52cy90b3JjaDMxMC9saWIvcHl0aG9uMy4xMC9zaXRlLXBhY2thZ2VzL2pheC9leHBlcmltZW50YWwvcGFsbGFzL29wcy90cHUvZmxhc2hfYXR0ZW50aW9uLnB5AGJvZHkAdmFsdWUAc3ltX25hbWUAX2ZsYXNoX2F0dGVudGlvbl9kcV9rZXJuZWwAZnVuY3Rpb25fdHlwZQB0cmFuc2Zvcm1faW5kaWNlcwB3aW5kb3dfYm91bmRzAC9tdWwAL2dldAAvYWRkAC9jb252ZXJ0X2VsZW1lbnRfdHlwZQAvc3ViAHByZWRpY2F0ZQAvY29uZAAtAC9zd2FwAC9zZWxlY3RfbgAvYnJvYWRjYXN0X2luX2RpbQAvZG90X2dlbmVyYWwAZGltZW5zaW9uAC9yZXBlYXQAdHJhbnNmb3JtXzAAdHJhbnNmb3JtXzEAdHJhbnNmb3JtXzIAdHJhbnNmb3JtXzMAdHJhbnNmb3JtXzQAdHJhbnNmb3JtXzUAdHJhbnNmb3JtXzYAdHJhbnNmb3JtXzcAa3ZfaW5kZXhfbWFwAC9ndAAvZXEAZW5kX29mX2t2X3NlcXVlbmNlAC9tYXNrZWRfbG9hZAB0cmFuc3Bvc2VfbGhzAHRyYW5zcG9zZV9yaHMAL2lvdGEAc3RhYmxlX21vc2FpYy52ZXJzaW9uAGRpbWVuc2lvbl9zZW1hbnRpY3MAaXRlcmF0aW9uX2JvdW5kcwBzY2FsYXJfcHJlZmV0Y2gAc2NyYXRjaF9vcGVyYW5kcwBtYWluAHdpbmRvd19wYXJhbXMAYmVsb3dfb3Jfb25fZGlhZwBfZmxhc2hfYXR0ZW50aW9uX2J3ZF9kcQBvdmVyZmxvd0ZsYWdzAC9zY2FuAHJ1bgAvbGUAL3BqaXQAZmFzdG1hdGgAdGltZXMAL2V4cAAvZGl2AHN0YXJ0X25ld19zZXF1ZW5jZQA=\", \"serialization_format\": 1, \"needs_layout_passes\": true}, \"implicit_sharding\": {\"type\": \"MANUAL\"}}"

      args = [q, k, v]
      if ab is not None:
        args += [ab]
      if segment_ids is not None:
        args += [q_segment_ids, kv_segment_ids]
      args += [expanded_l, expanded_m, grad_output, expanded_grad_i]

      outputs = [q]
      if ab is not None:
        outputs += [ab]
      grads = torch_xla._XLAC._xla_tpu_custom_call(args, payload,
                                                   [i.shape for i in outputs],
                                                   [i.dtype for i in outputs])
      if ctx.needs_input_grad[0]:
        grad_q = grads[0]
      if ctx.needs_input_grad[-3]:
        grad_ab = grads[1]

    if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
      payload = "{\"custom_call_config\": {\"body\": \"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BTM1AwNZEQMBBwdZDQMBBQU5AwN3DwMBAwN3BwMBEQd3TwMNBTc7IwZiAwMBA0EDA3sHAwERB3t1Aw0FQ0UlFHsDRwkD2XoDAwPbBwMBAwPhEQMBBwfhDQMBBVdZAwPlBwMBAwPnEQMBBwfnDQMBBV1fAwM9BQMDAwM9BQMDCwY9AwMDYQMDPQUDAwUGPQMTCwtjZWdpCQY9AxEDawMDPwUDAwMDPwUDAwsGPwMDA2EDAz8FAwMFBj8DEwsNb3FzdQkGPwMRA3cDA0EFAwMDA0EFAwMLBkEDAwNbAwNBBQMDBQZBAxMLCXt9f4EJBkEDEQODAwNDBQMDAwNDBQMDCwZDAwMDWwMDQwUDAwUGQwMdCw+HiYuNCQZDAwUDjwMDRQUDAwMDRQUDAwsGRQMDA1sDA0UFAwMFBkUDHQsRk5WXmQkGRQMFA5sDA0cFAwMDA0cFAwMLBkcDAwNbAwNHBQMDBQZHAxMLE5+ho6UJBkcDEQOnAwNJBQMDAwNJBQMDCwZJAwMDWwMDSQUDAwUGSQMdCxWrra+xCQZJAwUDswMD6+kDByEH6+0DBweFbbcrAz4EOgQDFwMD9REDAQcH9Q0DAQUHvRMHjw0DAQW/WxUGjwMXA8ETB48NAxcFu8MrA1oEVgQDFwMD+REDAQcH+Q0DAQUFyRMHkQ0DAQXLYRUGkQMXA80TB5ENAxcFx88DA5MPAwEDA5MHAwERB5OCBAMlBdHFAwMGApUDCQMDBgKWBAMJFQYOAgMHA9kVBg4CAwcD2x0GJgUDBwfX3d8nBy4FJwMHBbnhKQc6BZkDBwOdLQdGBScDBwXj5TMHSgUnAwcD5wMDnV4FAwkVBp0DBQPrNQedJwMFBe2RKQd2BZkDBwPvLweCBScDBwXp8TEHjgUeAgMHA/MfBpoFAx8D9QMDKgImAgMFIQcqAi4CAwUH96n5CwahAwMDYQMDoQUDAwUGoQMFBx39/ycHqgUnAwUFAgL7CwZnAwMDYQMDZwUDAwUGZwMFBx0KAg4CGQRnCQYCHQoCDgIDAzoC6QMHIQc6Au0DBwepeRYCKQfGBZkDBwO1LQfSBScDBwUaAh4CLwfWBScDBwUiAvMxB9oFHgIDBwMmAh8G5gUDHwMqAgMDPgImAgMFIQc+Ai4CAwUHLgKFMgILBqcDAwNhAwOnBQMDBQanAwUHGzoCPgInB/IFJwMFBUICNgILBmkDAwNhAwNpBQMDBQZpAwUHG0oCTgIZBGkJRgIbSgJOAgMD5Q8DAQMD2w8DARcA2QMBBRcA2QMDW24DAwEDA1sPAwEDA1sHAwERB1vNAw0FB0kjBoIDAwEDTwMDfQcDAREHfXUDDQVRUyUUfQNVCQMtZQMDgQUDAwMDgQUDAwUGgQMFBx1XWR8GmgMDEQNbAwMtBQMDAwMtBQMDAwMtBQMDAwMtBQMDBQYtAxMLGV9hY2UJBi0DEQNnCQYtAxMDXRkELQ1rGV9hY2UDA4UFAwMDA4UFAwMFBoUDBQcbbW8fBqYDAxEDcQMDLwUDAwMDLwUDAwMDLwUDAwMDLwUDAwUGLwMTCxd1d3l7CQYvAxEDfQkGLwMTA3MZBC8NgRd1d3l7FwDTAwEFFwDTDwABDREB4gIHAydDCQEBAQEBAQEBAwM3DwMBEwc3DQMBBQcJAwMTEQMBBwcTDQMBBQsNAwM5DwMBGwc5DQMBBQ8RAwMTEQMBBwcTDQMBBQUVAwMjDwMBAwMjBwMBEQcjTwMNBRMXAwM7BwMBHQY7AwEHHQcfAwMBBwMBAwMBBwMBDwQBCQEDISMNEQHmAgcDDQ8JAQEBAQEBAQEDAwEHAwEDAwEHAwEPBAEJAQMFCQ0RAeoCBwMNDwkBAQEBAQEBAQMDAQcDAQMDAQcDAQ8EAQkBAwUJDREB7gIHAw0PCQEBAQEBAQEBAwMBBwMBAwMBBwMBDwQBCQEDBwkNEQHyAgcDDQ8JAQEBAQEBAQEDAwEHAwEDAwEHAwEPBAEJAQMHCQ0RAfYCBwMnQwkBAQEBAQEBAQMDNw8DARMHNw0DAQUHCQMDExEDAQcHEw0DAQULDQMDOQ8DARsHOQ0DAQUPEQMDExEDAQcHEw0DAQUFFQMDIw8DAQMDIwcDAREHI08DDQUTFwMDOwcDAR0GOwMBBx0HHwMDAQcDAQMDAQcDAQ8EAQkBAyEjDREB+gIHAydDCQEBAQEBAQEBAwM3DwMBEwc3DQMBBQcJAwMTEQMBBwcTDQMBBQsNAwM5DwMBGwc5DQMBBQ8RAwMTEQMBBwcTDQMBBQUVAwMjDwMBAwMjBwMBEQcjTwMNBRMXAwM7BwMBHQY7AwEHHQcfAwMBBwMBAwMBBwMBDwQBCQEDISMNEQH+AgcDDQ8JAQEBAQEBAQEDAwEHAwEDAwEHAwEPBAEJAQMFCQ0RAQIDBwMNDwkBAQEBAQEBAQMDAQcDAQMDAQcDAQ8EAQkBAwUJBgMBBQEAchfLGQsLDRNrGy0xSwsNCR0zIx0LIyEjKS0nGxcNHR0PCQ0lCwkVCRsZGRkZGRkZGRkRJRUFDZsRGw0VCy0LOQsbHSUNHRMP6xcTIxcXExcXDxkXGxsXGxUjFxcZFSMlFxkfDw0JHRFidWlsdGluAHN0YWJsZV9tb3NhaWMAdHB1AGFyaXRoAG1vZHVsZQBhcml0aC5jb25zdGFudAB2ZWN0b3IubG9hZABhcml0aC5tdWxpAHZlY3Rvci5zaGFwZV9jYXN0AGFyaXRoLmluZGV4X2Nhc3QAZnVuYy5mdW5jAGZ1bmMucmV0dXJuAGFyaXRoLmNtcGkAYXJpdGguYWRkaQB2ZWN0b3IuYnJvYWRjYXN0AHNjZi55aWVsZAB2ZWN0b3Iuc3RvcmUAYXJpdGguc3ViaQBhcml0aC5zZWxlY3QAYXJpdGgudHJ1bmNmAHRwdS5tYXRtdWwAYXJpdGguZXh0dWkAc2NmLmlmAGFyaXRoLmFkZGYAdHB1LnJlcGVhdAB0cHUuaW90YQBhcml0aC5zdWJmAGFyaXRoLm11bGYAdmVjdG9yLnRyYW5zcG9zZQBtYXRoLmV4cABhcml0aC5kaXZmAC9ob21lL2JiYWhsL21pbmljb25kYTMvZW52cy90b3JjaDMxMC9saWIvcHl0aG9uMy4xMC9zaXRlLXBhY2thZ2VzL2pheC9leHBlcmltZW50YWwvcGFsbGFzL29wcy90cHUvZmxhc2hfYXR0ZW50aW9uLnB5AGtfYm9keQBzeW1fbmFtZQBmdW5jdGlvbl90eXBlAHZhbHVlAHRyYW5zZm9ybV9pbmRpY2VzAHdpbmRvd19ib3VuZHMAL21hc2tlZF9sb2FkAC9tdWwAX2ZsYXNoX2F0dGVudGlvbl9ka3Zfa2VybmVsAC9hZGQAL2NvbnZlcnRfZWxlbWVudF90eXBlAC9zdWIAcHJlZGljYXRlAC9zd2FwAC9kb3RfZ2VuZXJhbABmb3J3YXJkAC9ob21lL2JiYWhsL3RyYW5zZm9ybWVycy9zcmMvdHJhbnNmb3JtZXJzL21vZGVscy9taXh0cmFsL21vZGVsaW5nX21peHRyYWwucHkAL2NvbmQALQBkaW1lbnNpb24AL2Jyb2FkY2FzdF9pbl9kaW0AL3JlcGVhdAB0cmFuc2Zvcm1fMAB0cmFuc2Zvcm1fMQB0cmFuc2Zvcm1fMgB0cmFuc2Zvcm1fMwB0cmFuc2Zvcm1fNAB0cmFuc2Zvcm1fNQB0cmFuc2Zvcm1fNgB0cmFuc2Zvcm1fNwB0cmFuc2Zvcm1fOABxb19pbmRleF9tYXAAL2d0AC9zZWxlY3RfbgAvZXEAL2dldABlbmRfb2ZfcV9zZXF1ZW5jZQAvc2NhbgBydW4AcV9ib2R5AHRyYW5zcG9zZV9saHMAdHJhbnNwb3NlX3JocwAvaW90YQAvdHJhbnNwb3NlAC9tYXNrZWRfc3dhcABzdGFydF9uZXdfc2VxdWVuY2UAc3RhYmxlX21vc2FpYy52ZXJzaW9uAGRpbWVuc2lvbl9zZW1hbnRpY3MAaXRlcmF0aW9uX2JvdW5kcwBzY2FsYXJfcHJlZmV0Y2gAc2NyYXRjaF9vcGVyYW5kcwBtYWluAHdpbmRvd19wYXJhbXMAYmVsb3dfb3Jfb25fZGlhZwBfZmxhc2hfYXR0ZW50aW9uX2J3ZF9ka3YAb3ZlcmZsb3dGbGFncwAvbGUAL3BqaXQAYm9keQBfZmxhc2hfYXR0ZW50aW9uX2tlcm5lbF9zaW5nbGVfYmF0Y2gAX2ZsYXNoX2F0dGVudGlvbl9rZXJuZWwAX2ZsYXNoX2F0dGVudGlvbl9pbXBsAGNvbXB1dGVfbG9zcwAvaG9tZS9iYmFobC90cmFuc2Zvcm1lcnMvc3JjL3RyYW5zZm9ybWVycy90cmFpbmVyLnB5AGZhc3RtYXRoAHRpbWVzAC9leHAAL2RpdgBwZXJtdXRhdGlvbgA=\", \"serialization_format\": 1, \"needs_layout_passes\": true}, \"implicit_sharding\": {\"type\": \"MANUAL\"}}"

      grads = torch_xla._XLAC._xla_tpu_custom_call(args, payload,
                                                   [k.shape, v.shape],
                                                   [k.dtype, v.dtype])

    if ctx.needs_input_grad[1]:
      grad_k = grads[0]
    if ctx.needs_input_grad[2]:
      grad_v = grads[1]

    # SPMD integration
    if partition_spec is not None:
      grad_q = xs.disable_manual_sharding(
          grad_q, partition_spec, full_shape, mesh=mesh).global_tensor
      grad_k = xs.disable_manual_sharding(
          grad_k, partition_spec, full_shape, mesh=mesh).global_tensor
      grad_v = xs.disable_manual_sharding(
          grad_v, partition_spec, full_shape, mesh=mesh).global_tensor

    return grad_q, grad_k, grad_v, None, None, None, None, grad_ab, None, None

class Gmm(torch.autograd.Function):
    @staticmethod
    def _eager_gmm(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        group_sizes: torch.Tensor
    ) -> torch.Tensor:
        """
        For testing purpose.
        """
        start = 0
        out = []
        for i, size in enumerate(group_sizes):
            result = lhs[start:start + size, :] @ rhs[i, :, :]
            out.append(result)
            start += group_sizes[i]
        return torch.cat(out)


    @staticmethod
    def _eager_gmm_backward(grad_output, lhs, rhs, group_sizes):
        """
        For testing purpose.
        """
        grad_lhs = []
        grad_rhs = []
        start = 0
        for i, size in enumerate(group_sizes):
            grad_lhs.append(grad_output[start:start + size, :] @ rhs[i, :, :].transpose(-1, -2))
            grad_rhs.append(lhs[start:start + size, :].t() @ grad_output[start:start + size, :])
            start += size
        return torch.cat(grad_lhs), torch.stack(grad_rhs)
    
    @staticmethod
    def gmm_no_jax_gate1(lhs, rhs, group_sizes, tiling=(512, 512, 512)):
        # payload corresponds to input shapes:
        # lhs: bfloat16(16384, 4096), 
        # rhs: bfloat16(8, 4096, 14336)
        # group_sizes: int32(8,)
        from torch_xla.experimental.custom_kernel import _make_group_metadata
        m, k, n = lhs.shape[0], lhs.shape[1], rhs.shape[2]
        tm, tk, tn = min(tiling[0], m), min(tiling[1], k), min(tiling[2], n)
        preferred_element_type = lhs.dtype
        group_offsets, group_ids, m_tile_ids, num_tiles = _make_group_metadata(
            group_sizes=group_sizes,
            m=m,
            tm=tm,
            visit_empty_groups=False,
        )
        group_offset_torch = torch.tensor([0], dtype=torch.int32).to(lhs.device)
        payload = "{\"custom_call_config\": {\"body\": \"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AQEBAQsBBwEHAQ0BFQEdARUBHwEDAzkJAwEDAzkjAwEDAzkJAwELBzmHAxEFBRcXBqICAwEDHQMDWwkDAQsHW48DEQUfIRkUWwMjCQMLHQMDvwIEAxkVBr8DBQMzAwNRAwMDAwNRAwMDBQZRAwUHFTc5EQRRCTUVNzkTAL0DAQUTAL0DAz+qAgMBAwM/IwMBAwM/CQMBCwc/hwMRBQUlFwa+AgMBAysDA10JAwELB12PAxEFLS8ZFF0DMQkDa+EDAxcDAwMDAxcDAwMFBhcDDwcPMzUDAwsDAwMDAwsDAwMDAwsDAwMFBgsDIQkROTs9GwYLAw8DPwMDGQMDAwMDGQMDAwUGGQMFBxVDRQMDR5UDBR0HR5cDBQc3QUkfB52bAwUFR0sDAw0DAwMDAw0DAwMFBg0DBQcVT1ERBA0JTRVPUQkGowMDAwMHBqMDAQUJVQkGpwMDA1cHBqcDAQUHWQMDqSMDASEHqTcDAQVXXQkGrQMDA18HBq0DAQUHYQkGrwMDAwMHBq8DAQULZQMDs2IDAwElB7M3AwEFZ2knA3YDbgMDExUGtwMTA2shB7c3AxMFbW8VBkkDEwNbAwNJIwMBAwNJCQMBCwdJkgMDGwVxcxUGSwMTA2MDA0sjAwEDA0sJAwELB0ueAwMbBXF7KQamAwMbBXmBAwNnAwMDAwNnAwMDBQZnAwUHFYWHAwNpAwMDAwNpAwMDBQZpAw8HE4uNKwbeAwMFA48tBuIDAwUHg4mRLwbyAwMPA5MDA08DAwMDA08DAwMFBk8DDwcTl5kRBE8JlROXmRMAnwMjTQMDFwMDAwMDFwMDAwUGFwMPBw8zNQMDCwMDAwMDCwMDAwMDCwMDAwUGCwMhCRE5Oz0bBgsDDwM/AwMZAwMDAwMZAwMDBQYZAwUHFUNFAwNHlQMFHQdHlwMFBzdBSR8HnZsDBQVHSwMDDQMDAwMDDQMDAwUGDQMFBxVPUREEDQlNFU9REwCfDwABDREB8QcDFRMPAQEBAQEBCwEHAQcBDQEJBoMDAwMDBwaDAwEFCw8DAwEJAwEPBAEFEQUNEQHzBwMbHw8BAQEBAQELAQcBBwENAQkGNQMDAwMHBjUDAQUJDwMDNQMDAwcGNQMBBQ0TIwd2AjcDAQURFQMDAQkDAQ8EAQcXBQENEQH3BwMVEw8BAQEBAQELAQcBBwENAQkGeQMDAwMHBnkDAQULDwMDAQkDAQ8EAQURAQYDAQUBAL4SoRUlFQsJCQ0VCxMdHRstCx0tHRsJLR0LIyEjKS0FDQlrGRkZDQsdJQ8tFR0bDxMhmxENC+UbGxcXExcXFxcXJQ8ZIxUbGRUXIxkZHw8NCR0RYnVpbHRpbgBzdGFibGVfbW9zYWljAHRwdQBhcml0aABtb2R1bGUAYXJpdGguY29uc3RhbnQAdmVjdG9yLmxvYWQAbWVtcmVmLmxvYWQAYXJpdGguaW5kZXhfY2FzdABhcml0aC5jbXBpAGZ1bmMuZnVuYwBmdW5jLnJldHVybgB2ZWN0b3Iuc3RvcmUAc2NmLnlpZWxkAHZlY3Rvci5icm9hZGNhc3QAYXJpdGguZXh0dWkAc2NmLmlmAHZlY3Rvci5zaGFwZV9jYXN0AHRwdS5tYXRtdWwAYXJpdGguYWRkZgBhcml0aC5hZGRpAGFyaXRoLnN1YmkAYXJpdGgubXVsaQB0cHUuaW90YQBhcml0aC5hbmRpAGFyaXRoLmV4dGYAYXJpdGguc2VsZWN0AGFyaXRoLnRydW5jZgAvaG9tZS9iYmFobC9taW5pY29uZGEzL2VudnMvdG9yY2gzMTAvbGliL3B5dGhvbjMuMTAvc2l0ZS1wYWNrYWdlcy9qYXgvZXhwZXJpbWVudGFsL3BhbGxhcy9vcHMvdHB1L21lZ2FibG94L2dtbS5weQAvZ2V0AHZhbHVlAGZvcndhcmQAL2hvbWUvYmJhaGwvdHJhbnNmb3JtZXJzL3NyYy90cmFuc2Zvcm1lcnMvbW9kZWxzL21peHRyYWwvbW9kZWxpbmdfbWl4dHJhbC5weQBfZ2V0X3N0b3JlX21hc2sAc3ltX25hbWUAa2VybmVsAF9zdG9yZV9hY2N1bQBmdW5jdGlvbl90eXBlAHByZWRpY2F0ZQAvY29udmVydF9lbGVtZW50X3R5cGUAX2FjY3VtAHRyYW5zZm9ybV9pbmRpY2VzAHdpbmRvd19ib3VuZHMAL2FkZAAvc3dhcAB0cmFuc2Zvcm1fMAB0cmFuc2Zvcm1fMQB0cmFuc2Zvcm1fMgAvaG9tZS9iYmFobC90cmFuc2Zvcm1lcnMvc3JjL3RyYW5zZm9ybWVycy90cmFpbmVyLnB5AC9lcQAvY29uZAAtAHN0YWJsZV9tb3NhaWMudmVyc2lvbgBkaW1lbnNpb25fc2VtYW50aWNzAGl0ZXJhdGlvbl9ib3VuZHMAc2NhbGFyX3ByZWZldGNoAHNjcmF0Y2hfb3BlcmFuZHMAbWFpbgB3aW5kb3dfcGFyYW1zAG91dF90cmFuc2Zvcm1faW5kaWNlcwBnbW0AY29tcHV0ZV9sb3NzAHRyYWluaW5nX3N0ZXAAcmhzX3RyYW5zZm9ybV9pbmRpY2VzAG92ZXJmbG93RmxhZ3MAL3N1YgBsaHNfdHJhbnNmb3JtX2luZGljZXMAL2RvdF9nZW5lcmFsAHRyYW5zcG9zZV9saHMAdHJhbnNwb3NlX3JocwBmYXN0bWF0aAAvbXVsAGRpbWVuc2lvbgAvaW90YQAvZ2UAL2x0AC9hbmQAL3NlbGVjdF9uAC9icm9hZGNhc3RfaW5fZGltAF96ZXJvX2FjYwA=\", \"cost_estimate\": {\"flops\": 1924145348608, \"transcendentals\": 0, \"bytes_accessed\": 8808038400}, \"serialization_format\": 1, \"needs_layout_passes\": true}, \"implicit_sharding\": {\"type\": \"MANUAL\"}}"
        return torch_xla._XLAC._xla_tpu_custom_call([
            num_tiles, group_offsets, group_ids, m_tile_ids, group_offset_torch, lhs, rhs], payload, [torch.Size([m, n])], [preferred_element_type])[0]


    @staticmethod
    def gmm_no_jax_gate2(lhs, rhs, group_sizes, tiling=(512, 512, 512)):
        # payload corresponds to input shapes:
        # lhs: bfloat16(16384, 14336), 
        # rhs: bfloat16(8, 14336, 4096)
        # group_sizes: int32(8,)
        from torch_xla.experimental.custom_kernel import _make_group_metadata
        m, k, n = lhs.shape[0], lhs.shape[1], rhs.shape[2]
        tm, tk, tn = min(tiling[0], m), min(tiling[1], k), min(tiling[2], n)
        preferred_element_type = lhs.dtype
        group_offsets, group_ids, m_tile_ids, num_tiles = _make_group_metadata(
            group_sizes=group_sizes,
            m=m,
            tm=tm,
            visit_empty_groups=False,
        )
        group_offset_torch = torch.tensor([0], dtype=torch.int32).to(lhs.device)
        payload = "{\"custom_call_config\": {\"body\": \"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AQEBAQsBBwEHAQ0BFQEdARUBHwEDAzkJAwEDAzkjAwEDAzkJAwELBzmHAxEFBRcXBqICAwEDHQMDWwkDAQsHW48DEQUfIRkUWwMjCQMLHQMDvwIEAxkVBr8DBQMzAwNRAwMDAwNRAwMDBQZRAwUHFTc5EQRRCTUVNzkTAL0DAQUTAL0DAz+qAgMBAwM/IwMBAwM/CQMBCwc/hwMRBQUlFwa+AgMBAysDA10JAwELB12PAxEFLS8ZFF0DMQkDa+EDAxcDAwMDAxcDAwMFBhcDDwcPMzUDAwsDAwMDAwsDAwMDAwsDAwMFBgsDIQkROTs9GwYLAw8DPwMDGQMDAwMDGQMDAwUGGQMFBxVDRQMDR5UDBR0HR5cDBQc3QUkfB52bAwUFR0sDAw0DAwMDAw0DAwMFBg0DBQcVT1ERBA0JTRVPUQkGowMDAwMHBqMDAQUJVQkGpwMDA1cHBqcDAQUHWQMDqSMDASEHqTcDAQVXXQkGrQMDA18HBq0DAQUHYQkGrwMDAwMHBq8DAQULZQMDs2IDAwElB7M3AwEFZ2knA3YDbgMDExUGtwMTA2shB7c3AxMFbW8VBkkDEwNbAwNJIwMBAwNJCQMBCwdJkgMDGwVxcxUGSwMTA2MDA0sjAwEDA0sJAwELB0ueAwMbBXF7KQamAwMbBXmBAwNnAwMDAwNnAwMDBQZnAwUHFYWHAwNpAwMDAwNpAwMDBQZpAw8HE4uNKwbeAwMFA48tBuIDAwUHg4mRLwbyAwMPA5MDA08DAwMDA08DAwMFBk8DDwcTl5kRBE8JlROXmRMAnwMjTQMDFwMDAwMDFwMDAwUGFwMPBw8zNQMDCwMDAwMDCwMDAwMDCwMDAwUGCwMhCRE5Oz0bBgsDDwM/AwMZAwMDAwMZAwMDBQYZAwUHFUNFAwNHlQMFHQdHlwMFBzdBSR8HnZsDBQVHSwMDDQMDAwMDDQMDAwUGDQMFBxVPUREEDQlNFU9REwCfDwABDREB8QcDFRMPAQEBAQEBCwEHAQcBDQEJBoMDAwMDBwaDAwEFCw8DAwEJAwEPBAEFEQUNEQHzBwMbHw8BAQEBAQELAQcBBwENAQkGNQMDAwMHBjUDAQUJDwMDNQMDAwcGNQMBBQ0TIwd2AjcDAQURFQMDAQkDAQ8EAQcXBQENEQH3BwMVEw8BAQEBAQELAQcBBwENAQkGeQMDAwMHBnkDAQULDwMDAQkDAQ8EAQURAQYDAQUBAL4SoRUlFQsJCQ0VCxMdHRstCx0tHRsJLR0LIyEjKS0FDQlrGRkZDQsdJQ8tFR0bDxMhmxENC+UbGxcXExcXFxcXJQ8ZIxUbGRUXIxkZHw8NCR0RYnVpbHRpbgBzdGFibGVfbW9zYWljAHRwdQBhcml0aABtb2R1bGUAYXJpdGguY29uc3RhbnQAdmVjdG9yLmxvYWQAbWVtcmVmLmxvYWQAYXJpdGguaW5kZXhfY2FzdABhcml0aC5jbXBpAGZ1bmMuZnVuYwBmdW5jLnJldHVybgB2ZWN0b3Iuc3RvcmUAc2NmLnlpZWxkAHZlY3Rvci5icm9hZGNhc3QAYXJpdGguZXh0dWkAc2NmLmlmAHZlY3Rvci5zaGFwZV9jYXN0AHRwdS5tYXRtdWwAYXJpdGguYWRkZgBhcml0aC5hZGRpAGFyaXRoLnN1YmkAYXJpdGgubXVsaQB0cHUuaW90YQBhcml0aC5hbmRpAGFyaXRoLmV4dGYAYXJpdGguc2VsZWN0AGFyaXRoLnRydW5jZgAvaG9tZS9iYmFobC9taW5pY29uZGEzL2VudnMvdG9yY2gzMTAvbGliL3B5dGhvbjMuMTAvc2l0ZS1wYWNrYWdlcy9qYXgvZXhwZXJpbWVudGFsL3BhbGxhcy9vcHMvdHB1L21lZ2FibG94L2dtbS5weQAvZ2V0AHZhbHVlAGZvcndhcmQAL2hvbWUvYmJhaGwvdHJhbnNmb3JtZXJzL3NyYy90cmFuc2Zvcm1lcnMvbW9kZWxzL21peHRyYWwvbW9kZWxpbmdfbWl4dHJhbC5weQBfZ2V0X3N0b3JlX21hc2sAc3ltX25hbWUAa2VybmVsAF9zdG9yZV9hY2N1bQBmdW5jdGlvbl90eXBlAHByZWRpY2F0ZQAvY29udmVydF9lbGVtZW50X3R5cGUAX2FjY3VtAHRyYW5zZm9ybV9pbmRpY2VzAHdpbmRvd19ib3VuZHMAL2FkZAAvc3dhcAB0cmFuc2Zvcm1fMAB0cmFuc2Zvcm1fMQB0cmFuc2Zvcm1fMgAvaG9tZS9iYmFobC90cmFuc2Zvcm1lcnMvc3JjL3RyYW5zZm9ybWVycy90cmFpbmVyLnB5AC9lcQAvY29uZAAtAHN0YWJsZV9tb3NhaWMudmVyc2lvbgBkaW1lbnNpb25fc2VtYW50aWNzAGl0ZXJhdGlvbl9ib3VuZHMAc2NhbGFyX3ByZWZldGNoAHNjcmF0Y2hfb3BlcmFuZHMAbWFpbgB3aW5kb3dfcGFyYW1zAG91dF90cmFuc2Zvcm1faW5kaWNlcwBnbW0AY29tcHV0ZV9sb3NzAHRyYWluaW5nX3N0ZXAAcmhzX3RyYW5zZm9ybV9pbmRpY2VzAG92ZXJmbG93RmxhZ3MAL3N1YgBsaHNfdHJhbnNmb3JtX2luZGljZXMAL2RvdF9nZW5lcmFsAHRyYW5zcG9zZV9saHMAdHJhbnNwb3NlX3JocwBmYXN0bWF0aAAvbXVsAGRpbWVuc2lvbgAvaW90YQAvZ2UAL2x0AC9hbmQAL3NlbGVjdF9uAC9icm9hZGNhc3RfaW5fZGltAF96ZXJvX2FjYwA=\", \"cost_estimate\": {\"flops\": 1924145348608, \"transcendentals\": 0, \"bytes_accessed\": 8472494080}, \"serialization_format\": 1, \"needs_layout_passes\": true}, \"implicit_sharding\": {\"type\": \"MANUAL\"}}"
        return torch_xla._XLAC._xla_tpu_custom_call([
            num_tiles, group_offsets, group_ids, m_tile_ids, group_offset_torch, lhs, rhs], payload, [torch.Size([m, n])], [preferred_element_type])[0]

    @staticmethod
    def tgmm_no_jax_gate2(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        group_sizes: torch.Tensor,
        tiling: Tuple[int, int, int] = (512, 512, 512)
    ) -> torch.Tensor:
        # payload corresponds to following inputs:
        # lhs: bfloat16(14336, 16384)
        # rhs: bfloat16(16384, 4096)
        # group_sizes: int32(8,)
        from torch_xla.experimental.custom_kernel import _make_group_metadata
        k, m, n, num_groups = lhs.shape[0], lhs.shape[1], rhs.shape[1], group_sizes.shape[0]
        tm, tk, tn = min(tiling[0], m), min(tiling[1], k), min(tiling[2], n)
        preferred_element_type = lhs.dtype
        group_offsets, group_ids, m_tile_ids, num_tiles = _make_group_metadata(
            group_sizes=group_sizes,
            m=m,
            tm=tm,
            visit_empty_groups=True,
        )
        group_offset_torch = torch.tensor([0], dtype=torch.int32).to(lhs.device)
        payload = "{\"custom_call_config\": {\"body\": \"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AwEXBAMHFwMBBgMBBQEA7hO3FRMdHRsXGRUJKR0JDS0tHZsTCy0dCyMhIyktCwkJDQsbCQkJCRkZGSUNBQ0dJSEVCx0VEyENCy0PCQvlFxcjKRcXExclFRsbDxkbGRcVFRcZIxcjGR8PDQkdEWJ1aWx0aW4Ac3RhYmxlX21vc2FpYwB0cHUAYXJpdGgAbW9kdWxlAGFyaXRoLmNvbnN0YW50AG1lbXJlZi5sb2FkAGFyaXRoLmluZGV4X2Nhc3QAYXJpdGguY21waQB2ZWN0b3IuYnJvYWRjYXN0AHZlY3Rvci5sb2FkAGFyaXRoLmFkZGkAc2NmLnlpZWxkAGZ1bmMuZnVuYwBhcml0aC5zdWJpAGZ1bmMucmV0dXJuAGFyaXRoLnNlbGVjdABhcml0aC5leHR1aQBzY2YuaWYAYXJpdGgudHJ1bmNmAHZlY3Rvci5zdG9yZQBhcml0aC5vcmkAdmVjdG9yLnNoYXBlX2Nhc3QAYXJpdGgubXVsaQB0cHUuaW90YQBhcml0aC5hbmRpAGFyaXRoLmV4dGYAdHB1Lml0ZXJhdGlvbl9ib3VuZAB2ZWN0b3IudHJhbnNwb3NlAHRwdS5tYXRtdWwAYXJpdGguYWRkZgAvaG9tZS9iYmFobC9taW5pY29uZGEzL2VudnMvdG9yY2gzMTAvbGliL3B5dGhvbjMuMTAvc2l0ZS1wYWNrYWdlcy9qYXgvZXhwZXJpbWVudGFsL3BhbGxhcy9vcHMvdHB1L21lZ2FibG94L2dtbS5weQAvZ2V0AF9kbwBrZXJuZWwAL2NvbnZlcnRfZWxlbWVudF90eXBlAC9hZGQAdmFsdWUAX2dldF9zdG9yZV9tYXNrAHN5bV9uYW1lAHByZWRpY2F0ZQBmdW5jdGlvbl90eXBlAC9zdWIAL3NlbGVjdF9uAF9nZXRfZ3JvdXBfc2l6ZQB0cmFuc2Zvcm1faW5kaWNlcwB3aW5kb3dfYm91bmRzAC9jb25kAC0AL3N3YXAAL2Jyb2FkY2FzdF9pbl9kaW0AdHJhbnNmb3JtXzAAdHJhbnNmb3JtXzEAdHJhbnNmb3JtXzIAL2d0AC9lcQAvbmUAL29yAF9zdG9yZV9hY2N1bQAvbXVsAC9pb3RhAC9nZQAvbHQAL2FuZABzdGFibGVfbW9zYWljLnZlcnNpb24AZGltZW5zaW9uX3NlbWFudGljcwBpdGVyYXRpb25fYm91bmRzAHNjYWxhcl9wcmVmZXRjaABzY3JhdGNoX29wZXJhbmRzAG1haW4Ad2luZG93X3BhcmFtcwBvdXRfdHJhbnNmb3JtX2luZGljZXMAdGdtbQBiYWNrd2FyZAAvaG9tZS9iYmFobC90cmFuc2Zvcm1lcnMvc3JjL3RyYW5zZm9ybWVycy9tb2RlbHMvbWl4dHJhbC9tb2RlbGluZ19taXh0cmFsLnB5AG92ZXJmbG93RmxhZ3MAcmhzX3RyYW5zZm9ybV9pbmRpY2VzAGxoc190cmFuc2Zvcm1faW5kaWNlcwAvcGppdABkaW0AL251bV9wcm9ncmFtcwBtYWtlX2dyb3VwX21ldGFkYXRhAGdtbQBkaW1lbnNpb24AcGVybXV0YXRpb24AL3RyYW5zcG9zZQAvZG90X2dlbmVyYWwAdHJhbnNwb3NlX2xocwB0cmFuc3Bvc2VfcmhzAGZhc3RtYXRoAF96ZXJvX2FjYwA=\", \"cost_estimate\": {\"flops\": 1924145348608, \"transcendentals\": 0, \"bytes_accessed\": 8455716864}, \"serialization_format\": 1, \"needs_layout_passes\": true}, \"implicit_sharding\": {\"type\": \"MANUAL\"}}"

        return torch_xla._XLAC._xla_tpu_custom_call([
            num_tiles, group_offsets, group_ids, m_tile_ids, group_offset_torch,
            lhs.t(), rhs
        ], payload, [torch.Size([num_groups, k, n])], [preferred_element_type])[0]
    
    @staticmethod
    def tgmm_no_jax_gate1(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        group_sizes: torch.Tensor,
        tiling: Tuple[int, int, int] = (512, 512, 512)
    ) -> torch.Tensor:
        # payload corresponds to following inputs:
        # lhs: bfloat16(4096, 16384)
        # rhs: bfloat16(16384, 14336)
        # group_sizes: int32(8,)
        from torch_xla.experimental.custom_kernel import _make_group_metadata
        k, m, n, num_groups = lhs.shape[0], lhs.shape[1], rhs.shape[1], group_sizes.shape[0]
        tm, tk, tn = min(tiling[0], m), min(tiling[1], k), min(tiling[2], n)
        preferred_element_type = lhs.dtype
        group_offsets, group_ids, m_tile_ids, num_tiles = _make_group_metadata(
            group_sizes=group_sizes,
            m=m,
            tm=tm,
            visit_empty_groups=True,
        )
        group_offset_torch = torch.tensor([0], dtype=torch.int32).to(lhs.device)
        payload = "{\"custom_call_config\": {\"body\": \"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LwkDAwUGLwMBBQ0TFQfaAhUDAQURFQMDAwcDARcEAwcXAwEGAwEFAQDuE7cVEx0dGxcZFQkpHQkNLS0dCy0dCyMhIyktCwkJDQsbCQkJCZsTGRkZJQ0FDR0lIRULHRUTIQ0LLQ8JC+UXFyMpFxcTFyUVGxsPGRsZFxUVFxkjFyMZHw8NCR0RYnVpbHRpbgBzdGFibGVfbW9zYWljAHRwdQBhcml0aABtb2R1bGUAYXJpdGguY29uc3RhbnQAbWVtcmVmLmxvYWQAYXJpdGguaW5kZXhfY2FzdABhcml0aC5jbXBpAHZlY3Rvci5icm9hZGNhc3QAdmVjdG9yLmxvYWQAYXJpdGguYWRkaQBzY2YueWllbGQAZnVuYy5mdW5jAGFyaXRoLnN1YmkAZnVuYy5yZXR1cm4AYXJpdGguc2VsZWN0AGFyaXRoLmV4dHVpAHNjZi5pZgBhcml0aC50cnVuY2YAdmVjdG9yLnN0b3JlAGFyaXRoLm9yaQB2ZWN0b3Iuc2hhcGVfY2FzdABhcml0aC5tdWxpAHRwdS5pb3RhAGFyaXRoLmFuZGkAYXJpdGguZXh0ZgB0cHUuaXRlcmF0aW9uX2JvdW5kAHZlY3Rvci50cmFuc3Bvc2UAdHB1Lm1hdG11bABhcml0aC5hZGRmAC9ob21lL2JiYWhsL21pbmljb25kYTMvZW52cy90b3JjaDMxMC9saWIvcHl0aG9uMy4xMC9zaXRlLXBhY2thZ2VzL2pheC9leHBlcmltZW50YWwvcGFsbGFzL29wcy90cHUvbWVnYWJsb3gvZ21tLnB5AC9nZXQAX2RvAGtlcm5lbAAvY29udmVydF9lbGVtZW50X3R5cGUAL2FkZAB2YWx1ZQBfZ2V0X3N0b3JlX21hc2sAc3ltX25hbWUAcHJlZGljYXRlAGZ1bmN0aW9uX3R5cGUAL3N1YgAvc2VsZWN0X24AX2dldF9ncm91cF9zaXplAHRyYW5zZm9ybV9pbmRpY2VzAHdpbmRvd19ib3VuZHMAL2NvbmQALQAvc3dhcAAvYnJvYWRjYXN0X2luX2RpbQB0cmFuc2Zvcm1fMAB0cmFuc2Zvcm1fMQB0cmFuc2Zvcm1fMgBiYWNrd2FyZAAvaG9tZS9iYmFobC90cmFuc2Zvcm1lcnMvc3JjL3RyYW5zZm9ybWVycy9tb2RlbHMvbWl4dHJhbC9tb2RlbGluZ19taXh0cmFsLnB5AC9ndAAvZXEAL25lAC9vcgBfc3RvcmVfYWNjdW0AL211bAAvaW90YQAvZ2UAL2x0AC9hbmQAc3RhYmxlX21vc2FpYy52ZXJzaW9uAGRpbWVuc2lvbl9zZW1hbnRpY3MAaXRlcmF0aW9uX2JvdW5kcwBzY2FsYXJfcHJlZmV0Y2gAc2NyYXRjaF9vcGVyYW5kcwBtYWluAHdpbmRvd19wYXJhbXMAb3V0X3RyYW5zZm9ybV9pbmRpY2VzAHRnbW0Ab3ZlcmZsb3dGbGFncwByaHNfdHJhbnNmb3JtX2luZGljZXMAbGhzX3RyYW5zZm9ybV9pbmRpY2VzAC9waml0AGRpbQAvbnVtX3Byb2dyYW1zAG1ha2VfZ3JvdXBfbWV0YWRhdGEAZ21tAGRpbWVuc2lvbgBwZXJtdXRhdGlvbgAvdHJhbnNwb3NlAC9kb3RfZ2VuZXJhbAB0cmFuc3Bvc2VfbGhzAHRyYW5zcG9zZV9yaHMAZmFzdG1hdGgAX3plcm9fYWNjAA==\", \"cost_estimate\": {\"flops\": 1924145348608, \"transcendentals\": 0, \"bytes_accessed\": 8455716864}, \"serialization_format\": 1, \"needs_layout_passes\": true}, \"implicit_sharding\": {\"type\": \"MANUAL\"}}"
        return torch_xla._XLAC._xla_tpu_custom_call([
            num_tiles, group_offsets, group_ids, m_tile_ids, group_offset_torch,
            lhs.t(), rhs
        ], payload, [torch.Size([num_groups, k, n])], [preferred_element_type])[0]

    @staticmethod
    def gmm_backward_no_jax_gate2(grad, lhs, rhs, group_sizes, tiling=(512, 512, 512)):
        grad_lhs = Gmm.gmm_no_jax_gate1(grad, rhs.transpose(-1, -2), group_sizes, tiling)
        grad_rhs = Gmm.tgmm_no_jax_gate2(lhs.t(), grad, group_sizes, tiling)
        return grad_lhs, grad_rhs
    
    @staticmethod
    def gmm_backward_no_jax_gate1(grad, lhs, rhs, group_sizes, tiling=(512, 512, 512)):
        grad_lhs = Gmm.gmm_no_jax_gate2(grad, rhs.transpose(-1, -2), group_sizes, tiling)
        grad_rhs = Gmm.tgmm_no_jax_gate1(lhs.t(), grad, group_sizes, tiling)
        return grad_lhs, grad_rhs

    @staticmethod
    @xp.trace_me("gmm_forward")
    def forward(ctx, hidden_states: torch.Tensor, top_ks: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
        """
        Integrated with PyTorch/XLA Pallas gmm:

        lhs: [m, hidden_size]
        top_ks: [m, k]
        w1: [num_experts, hidden_size, ffn_dim]
        w2: [num_experts, ffn_dim, hidden_size]
        w3: [num_experts, hidden_size, ffn_dim]
        """
        from torch_xla.experimental.custom_kernel import _histogram, gmm

        device = hidden_states.device
        if device == torch.device('cpu'):
            gmm = Gmm._eager_gmm
        # m is global shape
        m, k, n, num_experts, l = hidden_states.shape[0], top_ks.shape[1], hidden_states.shape[-1], w1.shape[0], w1.shape[-1]

        # Create a new node to keep the original sharding spec.
        zero = torch.zeros((1,), device=device, dtype=hidden_states.dtype)
        full_w1 = w1 + zero
        full_w2 = w2 + zero
        full_w3 = w3 + zero

        # Enter manual sharding zone
        if xs.get_global_mesh() is not None:
            if SINGLE_SLICE:
                hidden_states = xs.enable_manual_sharding(hidden_states, ('fsdp', None)).global_tensor
                top_ks = xs.enable_manual_sharding(top_ks, ('fsdp', None)).global_tensor
                w1 = xs.enable_manual_sharding(full_w1, (None, None, 'tensor')).global_tensor
                w2 = xs.enable_manual_sharding(full_w2, (None, 'tensor', None)).global_tensor
                w3 = xs.enable_manual_sharding(full_w3, (None, None, 'tensor')).global_tensor
            else:
                hidden_states = xs.enable_manual_sharding(hidden_states, (('dcn', 'fsdp'), None)).global_tensor
                top_ks = xs.enable_manual_sharding(top_ks, (('dcn', 'fsdp'), None)).global_tensor
                w1 = xs.enable_manual_sharding(full_w1, (None, None, 'tensor')).global_tensor
                w2 = xs.enable_manual_sharding(full_w2, (None, 'tensor', None)).global_tensor
                w3 = xs.enable_manual_sharding(full_w3, (None, None, 'tensor')).global_tensor
        

        # We want to create one big batch of tokens that has all top-k choices in it.
        # Our tokens will thus be duplicated k-times in the batch. To do this we,
        # first flatten the expert choices list and argsort it. This gives us an array
        # of length B * K. We then create a tiled arange of size B * K and index
        # into the expert choices list. This will give us the set of indices we need
        # to gather from the xs to create this big batch.
        top_flat = top_ks.flatten()
        hidden_states_order = top_flat.argsort()
        hidden_states_reverse_order = hidden_states_order.argsort()
        # Always replicated, so okay to skip manual sharding.
        hidden_states_indices = torch.arange(hidden_states.shape[0], device=device).repeat_interleave(k)[hidden_states_order]
        hidden_states_sorted = hidden_states[hidden_states_indices]

        group_sizes = _histogram(top_flat.to(torch.int32), 0, num_experts - 1)

        # Replicated MixtralBlockSparseTop2MLP.forward
        # Here we just use silu and ignore the configuration given we need to manually write the backward pass.
        # gmm1 = Gmm.gmm_no_jax_gate1(hidden_states_sorted, w1, group_sizes)
        # gmm3 = Gmm.gmm_no_jax_gate1(hidden_states_sorted, w3, group_sizes)
        gmm1 = gmm(hidden_states_sorted, w1, group_sizes)
        gmm3 = gmm(hidden_states_sorted, w3, group_sizes)
         # Should I save silu activations?
        silu = F.silu(gmm1)
        sgmm = silu * gmm3
        # gmm2 = Gmm.gmm_no_jax_gate2(sgmm, w2, group_sizes)
        gmm2 = gmm(sgmm, w2, group_sizes)
        current_hidden_states = gmm2[hidden_states_reverse_order].reshape(-1, k, n)

        # Exit manual sharding zone
        if xs.get_global_mesh() is not None:
            # For 2D sharding, we need to manually reduce-scatter the final results
            mesh = xs.get_global_mesh()
            if mesh.shape()['tensor'] > 1:
                # Assume tensor axis is the last dim. Otherwise, we will need some complicated transoforms.
                assert mesh.get_axis_name_idx('tensor') == len(mesh.mesh_shape) - 1
                device_ids = mesh.get_logical_mesh()
                device_ids = device_ids.reshape(-1, device_ids.shape[-1])
                ctx.device_ids = device_ids

                # Only reduce-scatter along tensor axis.
                current_hidden_states = torch_xla.torch_xla._XLAC._xla_spmd_reduce_scatter(xm.REDUCE_SUM, current_hidden_states, 1.0, -1, device_ids.shape[-1], device_ids.tolist())

            if SINGLE_SLICE:
                current_hidden_states = xs.disable_manual_sharding(current_hidden_states, ('fsdp', None, 'tensor'), (m, k, n)).global_tensor
                hidden_states_sorted = xs.disable_manual_sharding(hidden_states_sorted, ('fsdp', None), (m * k, n)).global_tensor
                gmm1 = xs.disable_manual_sharding(gmm1, ('fsdp', 'tensor'), (m * k, l)).global_tensor
                gmm3 = xs.disable_manual_sharding(gmm3, ('fsdp', 'tensor'), (m * k, l)).global_tensor
                silu = xs.disable_manual_sharding(silu, ('fsdp', 'tensor'), (m * k, l)).global_tensor
                sgmm = xs.disable_manual_sharding(sgmm, ('fsdp', 'tensor'), (m * k, l)).global_tensor
            else:
                current_hidden_states = xs.disable_manual_sharding(current_hidden_states, (('dcn','fsdp'), None, 'tensor'), (m, k, n)).global_tensor
                # Checkpoints for backward
                hidden_states_sorted = xs.disable_manual_sharding(hidden_states_sorted, (('dcn', 'fsdp'), None), (m * k, n)).global_tensor
                gmm1 = xs.disable_manual_sharding(gmm1, (('dcn', 'fsdp'), 'tensor'), (m * k, l)).global_tensor
                gmm3 = xs.disable_manual_sharding(gmm3, (('dcn', 'fsdp'), 'tensor'), (m * k, l)).global_tensor
                silu = xs.disable_manual_sharding(silu, (('dcn', 'fsdp'), 'tensor'), (m * k, l)).global_tensor
                sgmm = xs.disable_manual_sharding(sgmm, (('dcn', 'fsdp'), 'tensor'), (m * k, l)).global_tensor

        # Save for backward
        ctx.save_for_backward(hidden_states_sorted, full_w1, full_w2, full_w3, gmm1, gmm3, silu, sgmm, hidden_states_order, hidden_states_reverse_order, group_sizes)
        ctx.k = k

        return current_hidden_states


    @staticmethod
    @xp.trace_me("gmm_backward")
    def backward(ctx, grad_output):
        from torch_xla.experimental.custom_kernel import _histogram, gmm_backward
        device = grad_output.device
        if device == torch.device('cpu'):
            gmm_backward = Gmm._eager_gmm_backward

        hidden_states_sorted, full_w1, full_w2, full_w3, gmm1, gmm3, silu, sgmm, hidden_states_order, hidden_states_reverse_order, group_sizes = ctx.saved_tensors
        m, k, n = grad_output.shape[0], ctx.k, hidden_states_sorted.shape[-1]

        # Create a new node to keep the original sharding spec.
        zero = torch.zeros((1,), device=device, dtype=hidden_states_sorted.dtype)
        hidden_states_sorted = hidden_states_sorted + zero
        gmm1 = gmm1 + zero
        gmm3 = gmm3 + zero
        silu = silu + zero
        sgmm = sgmm + zero

        # Enter manual sharding zone
        if xs.get_global_mesh() is not None:
            if SINGLE_SLICE:
                hidden_states_sorted = xs.enable_manual_sharding(hidden_states_sorted, ('fsdp', None)).global_tensor
            else:
                hidden_states_sorted = xs.enable_manual_sharding(hidden_states_sorted, (('dcn', 'fsdp'), None)).global_tensor
            w1 = xs.enable_manual_sharding(full_w1, (None, None, 'tensor')).global_tensor
            w2 = xs.enable_manual_sharding(full_w2, (None, 'tensor', None)).global_tensor
            w3 = xs.enable_manual_sharding(full_w3, (None, None, 'tensor')).global_tensor
            temp_sharding_spec = ('fsdp', 'tensor') if SINGLE_SLICE else (('dcn', 'fsdp'), 'tensor')
            gmm1 = xs.enable_manual_sharding(gmm1, temp_sharding_spec).global_tensor
            gmm3 = xs.enable_manual_sharding(gmm3, temp_sharding_spec).global_tensor
            silu = xs.enable_manual_sharding(silu, temp_sharding_spec).global_tensor
            sgmm = xs.enable_manual_sharding(sgmm, temp_sharding_spec).global_tensor
            if SINGLE_SLICE:
                grad_output = xs.enable_manual_sharding(grad_output, ('fsdp', None, None)).global_tensor
            else:
                grad_output = xs.enable_manual_sharding(grad_output, (('dcn', 'fsdp'), None, None)).global_tensor


        grad_output_sorted = grad_output.reshape(-1, n)[hidden_states_order]
        # grad_output, grad_w2 = Gmm.gmm_backward_no_jax_gate2(grad_output_sorted, sgmm, w2, group_sizes)
        grad_output, grad_w2 = gmm_backward(grad_output_sorted, sgmm, w2, group_sizes)

        grad_gmm1 = gmm3 * grad_output
        grad_gmm1 = torch.ops.aten.silu_backward(grad_gmm1, gmm1)
        
        # grad_gmm1, grad_w1 = Gmm.gmm_backward_no_jax_gate1(grad_gmm1, hidden_states_sorted, w1, group_sizes)
        grad_gmm1, grad_w1 = gmm_backward(grad_gmm1, hidden_states_sorted, w1, group_sizes)

        grad_gmm3 = silu * grad_output
        # grad_gmm3, grad_w3 = Gmm.gmm_backward_no_jax_gate1(grad_gmm3, hidden_states_sorted, w3, group_sizes)
        grad_gmm3, grad_w3 = gmm_backward(grad_gmm3, hidden_states_sorted, w3, group_sizes)

        grad_output = grad_gmm1 + grad_gmm3

        grad_output = grad_output[hidden_states_reverse_order]
        grad_output = grad_output.reshape(-1, k, grad_output.shape[-1]).sum(dim=1)
        # Exit manual sharding zone
        if xs.get_global_mesh() is not None:
            if not hasattr(ctx, "device_ids"):
                # Here we do a manual reduce scatter as SPMD will not be able to infer this after the manual sharding zone.
                if SINGLE_SLICE:
                    groups = [xs.get_global_mesh().device_ids]  # a single group across the whole world
                else:
                    groups = [list(range(0, 256)), list(range(256, 512))]
                world_size = len(groups[0])
                grad_w1 = torch_xla.torch_xla._XLAC._xla_spmd_reduce_scatter(xm.REDUCE_SUM, grad_w1, 1 / world_size, -1, world_size, groups)
                grad_w2 = torch_xla.torch_xla._XLAC._xla_spmd_reduce_scatter(xm.REDUCE_SUM, grad_w2, 1 / world_size, -2, world_size, groups)
                grad_w3 = torch_xla.torch_xla._XLAC._xla_spmd_reduce_scatter(xm.REDUCE_SUM, grad_w3, 1 / world_size, -1, world_size, groups)

                if SINGLE_SLICE:
                    grad_output = xs.disable_manual_sharding(grad_output, ('fsdp', None), (m, n)).global_tensor
                else:
                    grad_output = xs.disable_manual_sharding(grad_output, (('dcn', 'fsdp'), None), (m, n)).global_tensor
                # TODO: make the 0s more programmatic.
                if SINGLE_SLICE:
                    grad_w1 = xs.disable_manual_sharding(grad_w1, (None, None, 'fsdp'), w1.shape).global_tensor
                    grad_w2 = xs.disable_manual_sharding(grad_w2, (None, 'fsdp', None), w2.shape).global_tensor
                    grad_w3 = xs.disable_manual_sharding(grad_w3, (None, None, 'fsdp'), w3.shape).global_tensor
                else:
                    grad_w1 = xs.disable_manual_sharding(grad_w1, (None, None,  'fsdp'), w1.shape).global_tensor
                    grad_w2 = xs.disable_manual_sharding(grad_w2, (None,  'fsdp', None), w2.shape).global_tensor
                    grad_w3 = xs.disable_manual_sharding(grad_w3, (None, None, 'fsdp'), w3.shape).global_tensor
            else:  # 2d sharding
                device_ids = ctx.device_ids

                # Only reduce-scatter along tensor axis.
                grad_output = torch_xla.torch_xla._XLAC._xla_spmd_reduce_scatter(xm.REDUCE_SUM, grad_output, 1.0, -1, device_ids.shape[-1], device_ids.tolist())

                # Only reduce-scatter along fsdp axis.
                # TODO: support multi-slice.
                device_ids = device_ids.T
                world_size = device_ids.shape[-1]
                grad_w1 = torch_xla.torch_xla._XLAC._xla_spmd_reduce_scatter(xm.REDUCE_SUM, grad_w1, 1 / world_size, -2, world_size, device_ids.tolist())
                grad_w2 = torch_xla.torch_xla._XLAC._xla_spmd_reduce_scatter(xm.REDUCE_SUM, grad_w2, 1 / world_size, -1, world_size, device_ids.tolist())
                grad_w3 = torch_xla.torch_xla._XLAC._xla_spmd_reduce_scatter(xm.REDUCE_SUM, grad_w3, 1 / world_size, -2, world_size, device_ids.tolist())

                grad_output = xs.disable_manual_sharding(grad_output, ('fsdp', 'tensor'), (m, n)).global_tensor
                grad_w1 = xs.disable_manual_sharding(grad_w1, (None, 'fsdp', 'tensor'), full_w1.shape).global_tensor
                grad_w2 = xs.disable_manual_sharding(grad_w2, (None, 'tensor', 'fsdp'), full_w2.shape).global_tensor
                grad_w3 = xs.disable_manual_sharding(grad_w3, (None, 'fsdp', 'tensor'), full_w3.shape).global_tensor
        return grad_output, None, grad_w1, grad_w2, grad_w3


class MixtralGmmTop2MLP(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts

        self.w1 = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.ffn_dim))
        self.w2 = nn.Parameter(torch.empty(self.num_experts, self.ffn_dim, self.hidden_dim))
        self.w3 = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.ffn_dim))

        self.reset_parameters()

    # The followings are copied from https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L49
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        init.kaiming_uniform_(self.w3, a=math.sqrt(5))

    @xp.trace_me("MixtralGmmTop2MLP")
    def forward(self, hidden_states, top_ks):
        return Gmm.apply(hidden_states, top_ks, self.w1, self.w2, self.w3)


class MixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.static = config.static
        self.gmm = config.gmm
        self.gmm_stack = config.gmm_stack

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        if not self.gmm or self.gmm_stack:
            self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts)])
        else:
            self.experts = MixtralGmmTop2MLP(config)

        # Jitter parameters
        self.jitter_noise = config.router_jitter_noise

    @xp.trace_me("MixtralSparseMoeBlock")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        if not self.gmm and not self.gmm_stack:
            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )

            # One hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be sollicitated
            if not self.static:
                expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

            # Loop over all available experts in the model and perform the computation on each expert
            for expert_idx in range(self.num_experts):
                expert_layer = self.experts[expert_idx]
                if not self.static:
                    idx, top_x = torch.where(expert_mask[expert_idx])

                    # Index the correct hidden states and compute the expert hidden state for
                    # the current expert. We need to make sure to multiply the output hidden
                    # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                    current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)  # why not current_state = hidden_states[top_x]?
                    current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

                    # However `index_add_` only support torch tensors for indexing so we'll use
                    # the `top_x` tensor here.
                    final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
                else:
                    routing_weights_idx = routing_weights.masked_fill(selected_experts != expert_idx, 0.0).sum(dim=-1, keepdim=True)
                    current_hidden_states = expert_layer(hidden_states) * routing_weights_idx  # We can't mask the input as there is non-linearities in the expert layer.
                    final_hidden_states += current_hidden_states.to(hidden_states.dtype)
        elif self.gmm_stack:
            w1 = torch.stack([expert.w1.weight.t() for expert in self.experts])
            w2 = torch.stack([expert.w2.weight.t() for expert in self.experts])
            w3 = torch.stack([expert.w3.weight.t() for expert in self.experts])
            final_hidden_states = Gmm.apply(hidden_states, selected_experts, w1, w2, w3)
            final_hidden_states = (final_hidden_states * routing_weights[..., None]).sum(dim=1)
        else:
            final_hidden_states = self.experts(hidden_states, selected_experts)
            final_hidden_states = (final_hidden_states * routing_weights[..., None]).sum(dim=1)

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class MixtralDecoderLayer(nn.Module):
    def __init__(self, config: MixtralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MIXTRAL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.block_sparse_moe = MixtralSparseMoeBlock(config)
        self.input_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @xp.trace_me("MixtralDecoderLayer")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


MIXTRAL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MixtralConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Mixtral Model outputting raw hidden-states without any specific head on top.",
    MIXTRAL_START_DOCSTRING,
)
# Copied from transformers.models.mistral.modeling_mistral.MistralPreTrainedModel with Mistral->Mixtral
class MixtralPreTrainedModel(PreTrainedModel):
    config_class = MixtralConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MixtralDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


MIXTRAL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        output_router_logits (`bool`, *optional*):
            Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
            should not be returned during inference.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Mixtral Model outputting raw hidden-states without any specific head on top.",
    MIXTRAL_START_DOCSTRING,
)
# Copied from transformers.models.mistral.modeling_mistral.MistralModel with MISTRAL->MIXTRAL,Mistral->Mixtral
class MixtralModel(MixtralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MixtralDecoderLayer`]

    Args:
        config: MixtralConfig
    """

    def __init__(self, config: MixtralConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MixtralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Ignore copy
    @add_start_docstrings_to_model_forward(MIXTRAL_INPUTS_DOCSTRING)
    @xp.trace_me("MixtralModel")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = 0

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mixtral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class MixtralForCausalLM(MixtralPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MixtralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(MIXTRAL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    # Ignore copy
    @xp.trace_me("MixtralForCausalLM")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MixtralForCausalLM

        >>> model = MixtralForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        output_router_logits=False,
        **kwargs,
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "output_router_logits": output_router_logits,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The Mixtral Model transformer with a sequence classification head on top (linear layer).

    [`MixtralForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    MIXTRAL_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaForSequenceClassification with Llama->Mixtral, LLAMA->MIXTRAL
class MixtralForSequenceClassification(MixtralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MixtralModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(MIXTRAL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
