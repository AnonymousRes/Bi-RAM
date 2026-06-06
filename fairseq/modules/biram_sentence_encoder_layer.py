# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, Optional, Tuple, Union, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.physics.vector import vlatex
from torch.fft import fft, ifft
from fairseq import utils
from fairseq.modules.sequence_norm import SequenceNorm
from fairseq.modules.relative_positional_bias import SimpleRelativePositionalBias, RotaryRelativePositionalBias
# from causal_conv1d import causal_conv1d_fn
from fairseq.modules import (
    FairseqDropout
)


import torch


def apply_attention(query, key, value, attn_padding_mask=None):
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(query.size(-1))
    if attn_padding_mask is not None:
        scores = scores.masked_fill(attn_padding_mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    return torch.matmul(p_attn, value), p_attn


def bi_ram(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Bidirectional Recalibrated Aggregated Memory (Bi-RAM)

    Args:
        x:    (B, L, D)
        gate: (B, L, D)

    Returns:
        out:  (B, L, D)
    """
    logits = gate

    fwd_max = logits.cummax(dim=1).values
    fwd_exp = torch.exp(logits - fwd_max)
    fwd_norm = fwd_exp.cumsum(dim=1)
    fwd_weighted = (fwd_exp * x).cumsum(dim=1)
    out_fwd = fwd_weighted / (fwd_norm + 1e-6)

    x_rev = torch.flip(x, dims=[1])
    logits_rev = torch.flip(logits, dims=[1])
    bwd_max = logits_rev.cummax(dim=1).values
    bwd_exp = torch.exp(logits_rev - bwd_max)
    bwd_norm = bwd_exp.cumsum(dim=1)
    bwd_weighted = (bwd_exp * x_rev).cumsum(dim=1)
    out_bwd_rev = bwd_weighted / (bwd_norm + 1e-6)
    out_bwd = torch.flip(out_bwd_rev, dims=[1])

    out = out_fwd + out_bwd

    return out


class CAMA_v2(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads, dropout=0.1, block_count=0, block_size=0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.d_k = embedding_dim // num_attention_heads
        self.block_count = block_count
        self.block_size = block_size

        self.output_linear = nn.Linear(embedding_dim, embedding_dim)
        self.att_dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, self_attn_padding_mask=None):
        B, L, D = query.size()
        L_kv = key.size(1)

        q = query.view(B, L, self.num_attention_heads, self.d_k).transpose(1, 2)
        k = key.view(B, L_kv, self.num_attention_heads, self.d_k).transpose(1, 2)
        v = value.view(B, L_kv, self.num_attention_heads, self.d_k).transpose(1, 2)

        x, attn = apply_attention(q, k, v, self_attn_padding_mask)

        x = x.transpose(1, 2).contiguous().view(B, L, D)
        out = self.att_dropout(self.output_linear(x))

        return out, key, value, attn


class BiRAM_Layer(nn.Module):
    def __init__(self, embedding_dim: int = 128, b_count: int = 0, b_size: int = 0, dropout=0.5):
        super().__init__()

        assert b_count > 0
        assert b_size > 0

        self.block_count = b_count
        self.block_size = b_size
        self.embedding_dim = embedding_dim
        self.dropout_p = dropout

        self.overlap_size = min(self.block_size // 4, 16) * 2

        self.biram_w_l = nn.Linear(embedding_dim, embedding_dim)
        self.biram_w_d = nn.Dropout(p=self.dropout_p)
        self.fuse_w_l = nn.Linear(embedding_dim, embedding_dim)
        self.fuse_w_d = nn.Dropout(p=self.dropout_p)
        self.biram_norm = nn.LayerNorm(embedding_dim)
        self.main_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=embedding_dim,
                kernel_size=5,
                padding="same",
            ),
            nn.Dropout(self.dropout_p),
            nn.SiLU()
        )
        self.main_conv_norm = nn.LayerNorm(embedding_dim)

        self.over_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=embedding_dim,
                kernel_size=3,
                padding="same",
            ),
            nn.Dropout(self.dropout_p),
            nn.SiLU()
        )
        self.over_conv_norm = nn.LayerNorm(embedding_dim)

        self.linear_fusion = nn.Linear(3 * embedding_dim, embedding_dim)
        self.linear_fusion_d = nn.Dropout(p=self.dropout_p)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor):
        """
        x: [B, L, D]
        return: [B, block_count, D]
        """

        B, L, D = x.shape
        assert D == self.embedding_dim
        L_pad = self.block_count * self.block_size
        assert L <= L_pad, "Input length exceeds configured block capacity"

        a_gate = F.silu(self.biram_w_d(self.biram_w_l(x)))  # [B, L, D]
        fusion_factor = F.sigmoid(self.fuse_w_d(self.fuse_w_l(x)))
        ram_x = self.biram_norm(bi_ram(x, a_gate) * fusion_factor + x)
        if L < L_pad:
            ram_x = F.pad(ram_x, (0, 0, 0, L_pad - L)).contiguous()
        ram_x_blocks = ram_x.view(
            B * self.block_count,
            self.block_size,
            D,
        )
        ram_x_blocks = self.main_conv_norm(self.main_conv(ram_x_blocks.transpose(1, 2)).transpose(1, 2) + ram_x_blocks)
        ram_block_repr = torch.max(ram_x_blocks, dim=1)[0].view(
            B * self.block_count, 1, D
        )

        b2s_attn_out, _ = apply_attention(
            ram_block_repr,
            ram_x_blocks,
            ram_x_blocks,
        )
        b2s_seq = b2s_attn_out.squeeze(1).view(
            B, self.block_count, D
        )
        b2b_seq, _ = apply_attention(
            b2s_seq,
            b2s_seq,
            b2s_seq,
        )
        b2b_attn_out = b2b_seq.view(
            B * self.block_count,
            1,
            D,
        )

        x_overlap = F.pad(
            ram_x_blocks.contiguous().view(
                B, self.block_count * self.block_size, D
            ),
            (0, 0, 0, self.overlap_size // 2),
            mode='replicate'
        )[:, self.block_size - self.overlap_size // 2:, :]
        x_blocks_overlap = x_overlap.unfold(
            dimension=1,
            size=self.overlap_size,
            step=self.block_size
        )
        x_blocks_overlap = x_blocks_overlap.permute(
            0, 1, 3, 2
        ).contiguous().view(
            B * self.block_count,
            self.overlap_size,
            D
        )
        x_blocks_overlap = self.over_conv_norm(self.over_conv(x_blocks_overlap.transpose(1, 2)).transpose(1, 2) + x_blocks_overlap)

        b2o_attn_out, _ = apply_attention(
            b2b_attn_out,
            x_blocks_overlap,
            x_blocks_overlap
        )

        combined_feat = torch.cat(
            [b2s_attn_out, b2b_attn_out, b2o_attn_out],
            dim=-1,
        )
        fused = self.linear_fusion_d(self.linear_fusion(combined_feat))
        refined_feat = fused.view(
            B, self.block_count, D
        )
        refined_feat = self.norm(refined_feat)

        return refined_feat


class BiRAMSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """
    def __init__(
        self,
        embedding_dim: int = 768,
        num_attention_heads: int = 8,
        block_count=0,
        block_size=0,
        dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = 'relu',
        norm_type='layernorm',
        export: bool = False,
        init_fn: Callable = None
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.block_count = block_count
        self.block_size = block_size
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.activation_dropout_module = FairseqDropout(activation_dropout, module_name=self.__class__.__name__)

        # Initialize blocks
        self.activation_fn = activation_fn # silu
        self.activation_fn = utils.get_activation_fn(self.activation_fn)
        # self.self_attn = DynamicAdapterMultiheadAttention(self.embedding_dim, num_attention_heads, dropout, self.shared_generator)
        # self.self_attn = DynamicAdapterMultiheadAttention_a3(self.embedding_dim, num_attention_heads, dropout,
                                                          # self.shared_generator)
        # self.self_attn = CAMA(self.embedding_dim, num_attention_heads, dropout, block_count=self.block_count, block_size=self.block_size)
        self.self_attn = CAMA_v2(self.embedding_dim, num_attention_heads, dropout, block_count=self.block_count,
                              block_size=self.block_size)
        self.norm_type = norm_type
        self.self_attn_layer_norm = SequenceNorm(self.norm_type, self.embedding_dim, affine=True, export=export)
        self.fc1 = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim)
        self.final_layer_norm = SequenceNorm(self.norm_type, self.embedding_dim, affine=True, export=export)

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        x, key, value, attn = self.self_attn(
            query=x,
            key=k,
            value=v,
            self_attn_padding_mask=self_attn_padding_mask
        )
        x = self.dropout_module(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, key, value, attn
