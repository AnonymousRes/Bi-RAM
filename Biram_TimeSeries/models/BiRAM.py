import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# Positional Encodings, Norm & Utils
# ==========================================

class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding
    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "silu":
        return F.silu
    raise ValueError("activation should be relu/gelu/silu, not {}".format(activation))


class SequenceNorm(nn.Module):
    def __init__(self, norm_type, d_model):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == 'BatchNorm':
            self.norm = nn.BatchNorm1d(d_model)
        elif norm_type == 'LayerNorm' or norm_type == 'layernorm':
            self.norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"Unsupported normalization: {norm_type}")

    def forward(self, x):
        if self.norm_type == 'BatchNorm':
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2)
            return x
        else:
            return self.norm(x)


def apply_attention(query, key, value, attn_padding_mask=None):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    if attn_padding_mask is not None:
        scores = scores.masked_fill(attn_padding_mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


# ==========================================
# Bi-RAM Core Operations
# ==========================================

def bi_ram(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    logits = gate

    # 1. Forward Aggregation
    fwd_max = logits.cummax(dim=1).values
    fwd_exp = torch.exp(logits - fwd_max)
    fwd_norm = fwd_exp.cumsum(dim=1)
    fwd_weighted = (fwd_exp * x).cumsum(dim=1)
    out_fwd = fwd_weighted / (fwd_norm + 1e-6)

    # 2. Backward Aggregation
    x_rev = torch.flip(x, dims=[1])
    logits_rev = torch.flip(logits, dims=[1])
    bwd_max = logits_rev.cummax(dim=1).values
    bwd_exp = torch.exp(logits_rev - bwd_max)
    bwd_norm = bwd_exp.cumsum(dim=1)
    bwd_weighted = (bwd_exp * x_rev).cumsum(dim=1)
    out_bwd_rev = bwd_weighted / (bwd_norm + 1e-6)
    out_bwd = torch.flip(out_bwd_rev, dims=[1])

    # 3. Bidirectional Fusion
    out = out_fwd + out_bwd
    return out


class CAMA(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads, dropout=0.1, block_count=0, block_size=0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.d_k = embedding_dim // num_attention_heads
        self.block_count = block_count
        self.block_size = block_size

        self.alpha_k = nn.Parameter(torch.zeros(1))
        self.alpha_v = nn.Parameter(torch.zeros(1))
        self.delta_scale_k = nn.Parameter(torch.zeros(1))
        self.delta_scale_v = nn.Parameter(torch.zeros(1))

        self.k_norm = nn.LayerNorm(embedding_dim)
        self.v_norm = nn.LayerNorm(embedding_dim)
        self.d_norm = nn.LayerNorm(embedding_dim)
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

        attn_mean = attn.mean(dim=1)
        delta = self.d_norm(torch.matmul(attn_mean.transpose(1, 2), out))

        delta_k = delta * self.delta_scale_k
        delta_v = delta * self.delta_scale_v

        alpha_k = torch.sigmoid(self.alpha_k)
        alpha_v = torch.sigmoid(self.alpha_v)

        key = self.k_norm((1 - alpha_k) * key + alpha_k * delta_k)
        value = self.v_norm((1 - alpha_v) * value + alpha_v * delta_v)

        return out, key, value, attn


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
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=5, padding="same"),
            nn.Dropout(self.dropout_p),
            nn.SiLU()
        )
        self.main_conv_norm = nn.LayerNorm(embedding_dim)

        self.over_conv = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding="same"),
            nn.Dropout(self.dropout_p),
            nn.SiLU()
        )
        self.over_conv_norm = nn.LayerNorm(embedding_dim)

        self.linear_fusion = nn.Linear(3 * embedding_dim, embedding_dim)
        self.linear_fusion_d = nn.Dropout(p=self.dropout_p)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        L_pad = self.block_count * self.block_size

        a_gate = F.silu(self.biram_w_d(self.biram_w_l(x)))
        fusion_factor = F.sigmoid(self.fuse_w_d(self.fuse_w_l(x)))

        ram_x = self.biram_norm(bi_ram(x, a_gate) * fusion_factor + x)

        if L < L_pad:
            ram_x = F.pad(ram_x, (0, 0, 0, L_pad - L)).contiguous()

        ram_x_blocks = ram_x.view(B * self.block_count, self.block_size, D)
        ram_x_blocks = self.main_conv_norm(self.main_conv(ram_x_blocks.transpose(1, 2)).transpose(1, 2) + ram_x_blocks)

        ram_block_repr = torch.max(ram_x_blocks, dim=1)[0].view(B * self.block_count, 1, D)

        b2s_attn_out, _ = apply_attention(ram_block_repr, ram_x_blocks, ram_x_blocks)

        b2s_seq = b2s_attn_out.squeeze(1).view(B, self.block_count, D)
        b2b_seq, _ = apply_attention(b2s_seq, b2s_seq, b2s_seq)
        b2b_attn_out = b2b_seq.view(B * self.block_count, 1, D)

        x_overlap = F.pad(
            ram_x_blocks.contiguous().view(B, self.block_count * self.block_size, D),
            (0, 0, 0, self.overlap_size // 2),
            mode='replicate'
        )[:, self.block_size - self.overlap_size // 2:, :]

        x_blocks_overlap = x_overlap.unfold(dimension=1, size=self.overlap_size, step=self.block_size)
        x_blocks_overlap = x_blocks_overlap.permute(0, 1, 3, 2).contiguous().view(B * self.block_count,
                                                                                  self.overlap_size, D)
        x_blocks_overlap = self.over_conv_norm(
            self.over_conv(x_blocks_overlap.transpose(1, 2)).transpose(1, 2) + x_blocks_overlap)

        b2o_attn_out, _ = apply_attention(b2b_attn_out, x_blocks_overlap, x_blocks_overlap)

        combined_feat = torch.cat([b2s_attn_out, b2b_attn_out, b2o_attn_out], dim=-1)
        fused = self.linear_fusion_d(self.linear_fusion(combined_feat))
        refined_feat = fused.view(B, self.block_count, D)
        refined_feat = self.norm(refined_feat)

        return refined_feat


class BiRAMSentenceEncoderLayer(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 768,
            num_attention_heads: int = 8,
            block_count=0,
            block_size=0,
            dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = 'relu',
            norm_type='LayerNorm'
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.block_count = block_count
        self.block_size = block_size

        self.dropout_module = nn.Dropout(dropout)
        self.activation_dropout_module = nn.Dropout(activation_dropout)
        self.activation_fn = _get_activation_fn(activation_fn)

        self.self_attn = CAMA_v2(self.embedding_dim, num_attention_heads, dropout, block_count=self.block_count,
                              block_size=self.block_size)

        self.norm_type = norm_type
        self.self_attn_layer_norm = SequenceNorm(self.norm_type, self.embedding_dim)
        self.fc1 = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim)
        self.final_layer_norm = SequenceNorm(self.norm_type, self.embedding_dim)

    def forward(self, x, k, v, self_attn_padding_mask=None):
        residual = x
        x, key, value, attn = self.self_attn(query=x, key=k, value=v, self_attn_padding_mask=self_attn_padding_mask)
        x = self.dropout_module(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, key, value, attn


# ==========================================
# Regression / Classification Wrapper
# ==========================================

class BiRAMClassiregressor(nn.Module):
    def __init__(self, feat_dim, max_len, d_model=512,
                 n_heads=8, num_layers=3, num_classes=100,
                 dropout=0.0, pos_encoding='fixed', activation='gelu', freeze=False, ktimes=1,
                 normalization_layer='LayerNorm'):
        super(BiRAMClassiregressor, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        # 1. Input Projection & Positional Encoding
        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)
        self.input_norm = nn.LayerNorm(d_model)

        # 2. Block initialization
        self.block_count = math.ceil(math.log2(self.max_len)) * ktimes
        self.block_size = math.ceil(self.max_len / self.block_count)

        # 3. Bi-RAM Modules
        self.biraml = BiRAM_Layer(
            embedding_dim=self.d_model,
            b_count=self.block_count,
            b_size=self.block_size,
            dropout=dropout
        )

        self.layers = nn.ModuleList([
            BiRAMSentenceEncoderLayer(
                embedding_dim=self.d_model,
                num_attention_heads=self.n_heads,
                block_count=self.block_count,
                block_size=self.block_size,
                dropout=dropout,
                activation_dropout=dropout,
                activation_fn=activation,
                norm_type=normalization_layer
            ) for _ in range(num_layers)
        ])

        # 4. Output Module
        self.act = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Linear(d_model * max_len, num_classes)
        return output_layer

    def forward(self, x_enc, padding_masks=None):
        # [B, L, feat_dim] -> [L, B, feat_dim]
        inp = x_enc.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(self.d_model)
        inp = self.pos_enc(inp)
        inp = inp.permute(1, 0, 2)  # [B, L, d_model]
        inp = self.input_norm(inp)

        # Generate block keys and values via Bi-RAM Layer
        key = value = self.biraml(inp)

        # Pass through stacked Bi-RAM Sentence Encoder Layers
        x = inp
        for layer in self.layers:
            x, key, value, _ = layer(x=x, k=key, v=value)

        # Apply activation & dropout
        output = self.act(x)
        output = self.dropout(output)

        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)

        return output
