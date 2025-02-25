import torch
from torch import nn as nn
from torch.nn import functional as F
import math


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    https://arxiv.org/pdf/1810.12890

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class MLP(nn.Module):
    """Feed-Forward Network (MLP) block used in Transformers encoder"""

    def __init__(
        self,
        in_features_dim,
        hidden_features_dim,
        dropout_rate=0.0,
    ):
        super().__init__()
        hidden_features_dim = hidden_features_dim if hidden_features_dim is not None else in_features_dim
        self.MLP = nn.Sequential(
            nn.Linear(in_features_dim, hidden_features_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_features_dim, in_features_dim),
        )

        self.drop = nn.Dropout(dropout_rate, inplace=True) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.MLP(x))


class MultiHeadSelfAttention(nn.Module):
    """Implements multi-head self-attention mechanism with L2 normalization.
    https://arxiv.org/pdf/2406.11430.pdf

    This module splits the input into multiple heads, computes scaled dot-product attention,
    and merges the results followed by a final projection layer.

    Args:
        block_idx (int): Identifier for the transformer block (for debugging/logging)
        embed_dim (int): Dimension of input embeddings (default: 768)
        num_heads (int): Number of parallel attention heads (default: 12)
        attn_dropout_rate (float): Dropout probability for attention weights (default: 0.0)
        proj_dropout_rate (float): Dropout probability for output projection (default: 0.0)

    Raises:
        AssertionError: If embed_dim is not divisible by num_heads
    """

    def __init__(
        self,
        block_idx: int,
        embed_dim: int = 768,
        num_heads: int = 12,
        attn_dropout_rate: float = 0.0,
        proj_dropout_rate: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.block_idx = block_idx
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Initialize scaling parameters
        # Learnable scale factor with log-space initialization and clamping
        # https://arxiv.org/pdf/2406.11430.pdf
        self.base_scale = 1.0
        self.log_scale_factor = nn.Parameter(
            torch.full((1, num_heads, 1, 1), 4.0).log(),
            requires_grad=True,
        )
        self.max_log_scale = torch.log(torch.tensor(100.0)).item()

        # Query/Key/Value projection layers
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # Typically no bias for keys
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # Output projection and dropout
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout_rate, inplace=True) if proj_dropout_rate > 0 else nn.Identity()
        self.attention_dropout_rate = attn_dropout_rate

        self.caching, self.cached_k, self.cached_v = False, None, None

    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None

    def forward(self, x: torch.Tensor, attention_bias: torch.Tensor) -> torch.Tensor:
        """Forward pass for multi-head self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embed_dim)
            attention_bias (torch.Tensor): Bias tensor for attention scores with shape
                (batch_size, num_heads, seq_length, seq_length) or broadcastable shape

        Returns:
            torch.Tensor: Output tensor of same shape as input (batch_size, seq_length, embed_dim)
        """
        batch_size, seq_length, embed_dim = x.shape

        # Project inputs to query/key/value space
        query = self.query_proj(x)  # (B, L, E)
        key = self.key_proj(x)  # (B, L, E)
        value = self.value_proj(x)  # (B, L, E)

        # Reshape and reorder dimensions for multi-head computation
        # New shape: (batch_size, num_heads, seq_length, head_dim)
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply L2 normalization to query and key vectors
        # Scale factor is learned per-head and clamped to prevent explosion
        scale_factor = self.log_scale_factor.clamp(max=self.max_log_scale).exp()
        query = F.normalize(query, p=2, dim=-1) * scale_factor
        key = F.normalize(key, p=2, dim=-1)

        if self.caching:
            if self.cached_k is None:
                self.cached_k = key
                self.cached_v = value
            else:
                key = self.cached_k = torch.cat((self.cached_k, key), dim=2)
                value = self.cached_v = torch.cat((self.cached_v, value), dim=2)

        # Compute scaled dot-product attention
        attention_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.base_scale,
            attn_mask=attention_bias,
            dropout_p=self.attention_dropout_rate if self.training else 0.0,
        )

        # Merge attention heads and apply final projection
        attention_output = (
            attention_output.transpose(1, 2)  # (B, L, num_heads, head_dim)
            .contiguous()
            .view(batch_size, seq_length, embed_dim)
        )

        return self.proj_dropout(self.output_proj(attention_output))


class AdaptiveLayerNormBeforeHead(nn.Module):
    """
    Adaptive Layer Normalization Module with conditioning input applied before the transformer head.

    This module applies a learnable adaptive transformation to the layer normalization output
    using conditioning input. The conditioning input determines scaling and shifting parameters,
    which modify the normalized tensor accordingly.

    Args:
        embedding_dim (int): The dimensionality of the input embeddings.
        conditioning_dim (int): The dimensionality of the conditioning input.
        norm_layer (nn.Module): Normalization layer (e.g., nn.LayerNorm) to be used.
    """

    def __init__(self, embedding_dim: int, conditioning_dim: int, norm_layer_eps: float):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.conditioning_dim = conditioning_dim

        # Layer normalization without affine transformation (no learned parameters)
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=norm_layer_eps, elementwise_affine=False)

        # Adaptive transformation network: Uses conditioning input to generate scale and shift parameters
        self.adaptive_linear = nn.Sequential(
            nn.SiLU(inplace=False),
            nn.Linear(conditioning_dim, 2 * embedding_dim),  # Predicts both scale and shift parameters
        )

    def forward(self, input_tensor: torch.Tensor, conditioning_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for adaptive layer normalization.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).
            conditioning_tensor (torch.Tensor): Conditioning tensor of shape (batch_size, conditioning_dim).

        Returns:
            torch.Tensor: Adaptively normalized output of shape (batch_size, seq_len, embedding_dim).
        """
        # Compute scale and shift parameters from conditioning input
        scale, shift = self.adaptive_linear(conditioning_tensor).view(-1, 1, 2, self.embedding_dim).unbind(2)

        # Apply layer normalization, then scale and shift transformation
        return self.layer_norm(input_tensor).mul(scale.add(1)).add_(shift)


class AdaptiveLayerNormSelfAttention(nn.Module):
    """Transformer block with Adaptive Layer Normalization (AdaLN) and self-attention.
    https://arxiv.org/pdf/1703.06868.pdf + Standard Transformer Decoder

    This module combines multi-head self-attention with MLP, using adaptive layer normalization
    parameters conditioned on an external input. Supports both shared and learned adaptation parameters.

    Args:
        block_idx (int): Identifier for the transformer block (for debugging/logging)
        embed_dim (int): Dimension of input embeddings
        cond_dim (int): Dimension of conditioning input
        norm_layer (nn.Module): Layer normalization class to use
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim (default: 4.0)
        dropout_rate (float): Dropout probability for MLP and attention outputs (default: 0.0)
        attn_dropout_rate (float): Dropout probability for attention weights (default: 0.0)
        drop_path (float): Stochastic depth drop probability (default: 0.0)

    Input Shapes:
        - x: (batch_size, seq_len, embed_dim)
        - cond_BD: (batch_size, cond_dim) or (batch_size, 1, cond_dim)
        - attn_bias: (batch_size, num_heads, seq_len, seq_len) or broadcastable shape

    Output Shape:
        - (batch_size, seq_len, embed_dim)
    """

    def __init__(
        self,
        block_idx: int,
        embed_dim: int,
        cond_dim: int,
        norm_layer_eps: float,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
        attn_dropout_rate: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.block_idx = block_idx
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim

        # Initialize core components
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.attention = MultiHeadSelfAttention(
            block_idx=block_idx,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout_rate=attn_dropout_rate,
            proj_dropout_rate=dropout_rate,
        )
        self.mlp = MLP(
            in_features_dim=embed_dim,
            hidden_features_dim=int(embed_dim * mlp_ratio),
            dropout_rate=dropout_rate,
        )

        # Layer normalization without learnable parameters
        self.layer_norm = nn.LayerNorm(embed_dim, eps=norm_layer_eps, elementwise_affine=False)

        # Adaptive layer normalization parameters
        # Block-specific learned transformation
        self.conditional_mlp = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(cond_dim, 6 * embed_dim))

    def forward(
        self,
        x: torch.Tensor,
        conditioning_tensor: torch.Tensor,
        attention_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass with adaptive layer normalization.

        Args:
            x: Input tensor of shape (B, L, C)
            conditioning_tensor: Conditioning input of shape (B, D) or (B, 1, D)
            attention_bias: Optional attention bias tensor for caucal training

        Returns:
            torch.Tensor: Transformed output of same shape as input
        """

        # print("debug adlnsa 0")

        # print("conditioning_tensor shape", conditioning_tensor.shape)
        # print("conditioning dim", self.cond_dim)

        # Generate parameters through learned transformation
        params = self.conditional_mlp(conditioning_tensor)

        # print("params shape", params.shape)

        params = params.view(-1, 1, 6, self.embed_dim)

        # print("debug adlnsa 1")

        attn_gamma, mlp_gamma, attn_scale, mlp_scale, attn_shift, mlp_shift = torch.split(
            params, split_size_or_sections=1, dim=2
        )

        # print("debug adlnsa 2")

        attn_gamma, mlp_gamma, attn_scale, mlp_scale, attn_shift, mlp_shift = (
            attn_gamma.squeeze(2),
            mlp_gamma.squeeze(2),
            attn_scale.squeeze(2),
            mlp_scale.squeeze(2),
            attn_shift.squeeze(2),
            mlp_shift.squeeze(2),
        )

        # print("debug adlnsa 3")

        # Attention branch with adaptive LN
        attention_output = self.layer_norm(x)
        attention_output = attention_output * (attn_scale + 1) + attn_shift
        attention_output = self.attention(attention_output, attention_bias) * attn_gamma
        x = x + self.drop_path(attention_output)

        # print("debug adlnsa 4")

        # MLP branch with adaptive LN
        mlp_output = self.layer_norm(x)
        mlp_output = mlp_output * (mlp_scale + 1) + mlp_shift
        mlp_output = self.mlp(mlp_output) * mlp_gamma
        x = x + self.drop_path(mlp_output)

        # print("debug adlnsa 5")

        return x


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    batch_size, seq_len, embed_dim, cond_dim, num_heads = 2, 8, 16, 4, 4
    dropout_rate, attn_dropout_rate, drop_path_rate = 0.1, 0.1, 0.1
    norm_layer_eps = 1e-6

    # Create test tensors
    x = torch.randn(batch_size, seq_len, embed_dim)
    conditioning_tensor = torch.randn(batch_size, cond_dim)
    attn_bias = torch.randn(batch_size, num_heads, seq_len, seq_len)

    # Test DropPath
    drop_path_layer = DropPath(drop_prob=0.1)
    drop_path_output = drop_path_layer(x)
    assert drop_path_output.shape == x.shape, "DropPath output shape mismatch"
    # print("DropPath passed.")

    # Test MLP
    mlp_layer = MLP(embed_dim, embed_dim * 4, dropout_rate)
    mlp_output = mlp_layer(x)
    assert mlp_output.shape == x.shape, "MLP output shape mismatch"
    # print("MLP passed.")

    # Test MultiHeadSelfAttention
    attn_layer = MultiHeadSelfAttention(
        block_idx=0,
        embed_dim=embed_dim,
        num_heads=num_heads,
        attn_dropout_rate=attn_dropout_rate,
        proj_dropout_rate=dropout_rate,
    )
    attn_output = attn_layer(x, attn_bias)
    assert attn_output.shape == x.shape, "MultiHeadSelfAttention output shape mismatch"
    # print("MultiHeadSelfAttention passed.")

    # Test AdaptiveLayerNormBeforeHead
    adaptive_norm = AdaptiveLayerNormBeforeHead(embed_dim, cond_dim, norm_layer_eps)
    adaptive_output = adaptive_norm(x, conditioning_tensor)
    assert adaptive_output.shape == x.shape, "AdaptiveLayerNormBeforeHead output shape mismatch"
    # print("AdaptiveLayerNormBeforeHead passed.")

    # Test AdaptiveLayerNormSelfAttention
    ada_attn_layer = AdaptiveLayerNormSelfAttention(
        block_idx=0,
        embed_dim=embed_dim,
        cond_dim=cond_dim,
        norm_layer_eps=norm_layer_eps,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        attn_dropout_rate=attn_dropout_rate,
        drop_path=drop_path_rate,
    )
    ada_attn_output = ada_attn_layer(x, conditioning_tensor, attn_bias)
    assert ada_attn_output.shape == x.shape, "AdaptiveLayerNormSelfAttention output shape mismatch"
    # print("AdaptiveLayerNormSelfAttention passed.")

    # print("All tests passed successfully!")


if __name__ == "__main__":
    main()
