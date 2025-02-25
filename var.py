import torch
from torch import nn as nn
from torch.nn import functional as F
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


from var_block import AdaptiveLayerNormBeforeHead, AdaptiveLayerNormSelfAttention
from multiscale_vqvae import VQVAE


from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.utils as vutils


def sample_with_top_k_and_top_p(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    generator: torch.Generator = None,
    num_samples: int = 1,
) -> torch.Tensor:
    """
    Samples from logits using combined top-k and nucleus (top-p) filtering.

    Applies in sequence:
    1. Top-k filtering: Keeps only the top 'k' highest probability tokens
    2. Nucleus filtering: Keeps the smallest set of tokens whose cumulative probability ≥ top_p
    3. Multinomial sampling from the filtered distribution

    Args:
        logits: Input tensor of shape (batch_size, seq_length, vocab_size)
        top_k: Number of top tokens to keep (disabled if <= 0)
        top_p: Cumulative probability threshold (disabled if <= 0.0)
        generator: Random number generator for reproducibility
        num_samples: Number of samples to draw. Use negative values for sampling without replacement.

    Returns:
        Sampled token indices of shape (batch_size, seq_length, num_samples)

    Note:
        - The sign of num_samples determines replacement: positive for with replacement, negative without
        - Top-k and top-p filters are applied in sequence (first top-k, then top-p)
    """
    batch_size, seq_length, vocab_size = logits.shape

    # Apply top-k filtering
    if top_k > 0:
        # Get the k-th largest logit value for each position
        topk_values = logits.topk(top_k, dim=-1, largest=True, sorted=False)[0]
        k_threshold = topk_values.amin(dim=-1, keepdim=True)  # (batch_size, seq_length, 1)

        # Create mask for logits below the k-th largest
        mask = logits < k_threshold
        logits.masked_fill_(mask, -torch.inf)

    # Apply nucleus (top-p) filtering
    if top_p > 0.0:
        # Sort logits in ascending order for cumulative probability calculation
        sorted_logits, sorted_indices = logits.sort(dim=-1, descending=False)
        sorted_probs = sorted_logits.softmax(dim=-1)

        # Calculate cumulative probabilities and create removal mask
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        sorted_mask = cumulative_probs <= (1 - top_p)

        # Ensure we keep at least one token (disable mask for last token)
        sorted_mask[..., -1:] = False

        # Scatter the sorted mask back to original indices
        original_mask = sorted_mask.scatter(dim=sorted_indices.ndim - 1, index=sorted_indices, src=sorted_mask)
        logits.masked_fill_(original_mask, -torch.inf)

    # Prepare sampling parameters
    replacement = num_samples >= 0
    actual_samples = abs(num_samples)
    probs = logits.softmax(dim=-1)

    # Reshape for multinomial sampling (batch_size * seq_length, vocab_size)
    flat_probs = probs.view(-1, vocab_size)

    # Sample from filtered distribution
    samples = torch.multinomial(
        flat_probs,
        num_samples=actual_samples,
        replacement=replacement,
        generator=generator,
    )

    # Reshape to (batch_size, seq_length, num_samples)
    return samples.view(batch_size, seq_length, actual_samples)


def gumbel_softmax_with_generator(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """
    Applies the Gumbel-Softmax trick with custom random number generator.

    Implements the reparameterization trick for categorical sampling, allowing
    differentiable samples through a temperature-controlled softmax.

    Args:
        logits: Input tensor of unnormalized log probabilities
        tau: Temperature parameter (lower values produce more discrete samples)
        hard: Return hard one-hot vectors when True, soft probabilities otherwise
        eps: Small epsilon for numerical stability (not used in this implementation)
        dim: Dimension along which to compute softmax
        generator: Random number generator for Gumbel noise

    Returns:
        Sampled tensor of same shape as input, either soft probabilities or
        straight-through hard one-hot vectors

    Note:
        When generator is None, falls back to standard F.gumbel_softmax
    """
    if generator is None:
        return F.gumbel_softmax(logits=logits, tau=tau, hard=hard, eps=eps, dim=dim)

    # Generate Gumbel noise using exponential distribution
    uniform_noise = torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
    gumbel_noise = -uniform_noise.exponential_(generator=generator).log()

    # Apply Gumbel trick and temperature scaling
    temperature_scaled = (logits + gumbel_noise) / tau
    soft_samples = temperature_scaled.softmax(dim)

    if not hard:
        return soft_samples

    # Straight-through estimator for hard samples
    # Get indices of maximum values
    hard_indices = soft_samples.argmax(dim=dim, keepdim=True)

    # Create one-hot encoding
    hard_samples = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(
        dim, hard_indices, 1.0
    )

    # Maintain gradient flow using straight-through estimator
    return hard_samples - soft_samples.detach() + soft_samples


class VAR(nn.Module):
    def __init__(
        self,
        vqvae: VQVAE,
        num_classes=1000,
        depth=16,
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4.0,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        dropout_path_rate=0.0,
        cond_dropout_rate=0.1,
        norm_eps=1e-6,
        resolutions_list=(1, 2, 3, 4, 5, 6, 8, 10, 12, 16),
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.vqvae_reference = vqvae
        self.vqvae_quantizer_reference = vqvae.vector_quantizer
        self.num_codebook_vectors = self.vqvae_quantizer_reference.num_codebook_vectors
        self.codebook_vectors_dim = self.vqvae_quantizer_reference.codebook_vectors_dim

        self.num_classes = num_classes
        self.depth = depth
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.cond_dim = embed_dim

        self.resolutions_list = resolutions_list
        self.scale_count = len(resolutions_list)

        self.cond_dropout_rate = cond_dropout_rate

        self.total_tokens = sum(ps**2 for ps in self.resolutions_list)
        self.start_token_count = self.resolutions_list[0] ** 2

        # Precompute the start/end token indices for each stage (0,1) (1,4) (4,16) etc.
        self.stage_token_ranges = []
        current_offset = 0
        for _, ps in enumerate(self.resolutions_list):
            self.stage_token_ranges.append((current_offset, current_offset + ps**2))
            current_offset += ps**2

        self.random_generator = torch.Generator(device="cuda")

        # Token embeddings
        self.token_embedding = nn.Linear(self.codebook_vectors_dim, self.embed_dim)

        # Class embedding for conditional generation
        init_std = math.sqrt(1.0 / self.embed_dim / 3)
        self.uniform_class_prob = torch.full(
            (1, self.num_classes),
            fill_value=1.0 / self.num_classes,
            dtype=torch.float32,
            device="cuda",
        )
        self.class_embedding = nn.Embedding(self.num_classes + 1, self.embed_dim)
        nn.init.trunc_normal_(self.class_embedding.weight.data, mean=0.0, std=init_std)

        # A position embedding for the start token
        self.pos_emb_start_token = nn.Parameter(
            torch.empty(1, self.start_token_count, self.embed_dim)  # (1, start_token_count, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_emb_start_token.data, mean=0.0, std=init_std)

        # Absolute position embeddings for all stages, concatenated
        pos_emb_list = []
        for ps in self.resolutions_list:
            pe = torch.empty(1, ps * ps, self.embed_dim)
            nn.init.trunc_normal_(pe, mean=0.0, std=init_std)
            pos_emb_list.append(pe)
        # Shape: [1, total_tokens, embed_dim]
        self.pos_emb_all_stages = nn.Parameter(torch.cat(pos_emb_list, dim=1))

        # Level (stage) embedding to distinguish patches from different stages
        self.stage_level_embedding = nn.Embedding(
            len(self.resolutions_list), self.embed_dim  # (scale_count, embed_dim)
        )
        nn.init.trunc_normal_(self.stage_level_embedding.weight.data, mean=0.0, std=init_std)

        # Build the stack of Transformer blocks
        self.dropout_path_rate = dropout_path_rate
        dropout_path_rates = [x.item() for x in torch.linspace(0, dropout_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                AdaptiveLayerNormSelfAttention(
                    block_idx=block_idx,
                    embed_dim=self.embed_dim,
                    cond_dim=self.cond_dim,
                    norm_layer_eps=norm_eps,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_rate=dropout_rate,
                    attn_dropout_rate=attn_dropout_rate,
                    drop_path=dropout_path_rates[block_idx],
                )
                for block_idx in range(depth)
            ]
        )

        # Create a causal mask for training (future tokens are masked)
        stage_index_tensor = torch.cat(
            [torch.full((ps * ps,), stage_idx) for stage_idx, ps in enumerate(self.resolutions_list)]
        ).view(1, self.total_tokens, 1)
        stage_index_tensor_transposed = stage_index_tensor.transpose(
            1, 2
        )  # [1, 1, total_tokens] [[[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]]] eg

        # Save a stage index map as buffer
        self.register_buffer("stage_indices", stage_index_tensor_transposed[:, 0].contiguous())  # [1, total_tokens]

        # Where stage_index_tensor >= stage_index_tensor_transposed => 0, else -inf
        # This ensures each token can attend only to the same or previous stages (no peeking ahead).
        causal_mask = torch.where(stage_index_tensor >= stage_index_tensor_transposed, 0.0, float("-inf")).reshape(
            1, 1, self.total_tokens, self.total_tokens
        )

        self.register_buffer("causal_attn_bias", causal_mask.contiguous())

        # Final projection head
        self.head_pre_norm = AdaptiveLayerNormBeforeHead(self.embed_dim, self.cond_dim, norm_layer_eps=norm_eps)
        self.output_head = nn.Linear(self.embed_dim, self.num_codebook_vectors)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        condition_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply final layer normalization and project to codebook logits.

        Args:
            hidden_states: Final hidden states of shape (B, L, embed_dim).
            condition_embeddings: Conditioning embeddings of shape (B, embed_dim).

        Returns:
            logits: Unnormalized output logits of shape (B, L, num_codebook_vectors).
        """
        # Apply adaptive layer norm with conditioning
        normalized_hidden_states = self.head_pre_norm(hidden_states.float(), condition_embeddings.float())

        # Project normalized hidden states to codebook logits
        logits = self.output_head(normalized_hidden_states).float()
        return logits

    def forward(
        self,
        class_labels: torch.LongTensor,
        teacher_forcing_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the VAR model.

        Args:
            class_labels: Class label tensor
            teacher_forcing_tokens: Teacher forcing input of shape
                (B, total_tokens-start_token_count, codebook_vec_dim).

        Returns:
            logits: Output logits of shape (B, total_tokens, num_codebook_vectors).
        """
        # For this model, we process tokens from stage 0 to stage N
        batch_size = teacher_forcing_tokens.shape[0]

        # Apply conditional dropout to labels (if cond_dropout_rate > 0)
        cond_dropout_mask = torch.rand(batch_size, device=class_labels.device) < self.cond_dropout_rate
        class_labels = torch.where(cond_dropout_mask, self.num_classes, class_labels)

        # Get class embeddings
        condition_embeddings = self.class_embedding(class_labels)  # shape: (batchsize, embed_dim)

        # Create "start-of-stage" embeddings for the first stage tokens
        first_stage_embed = condition_embeddings.unsqueeze(1).expand(
            batch_size, self.start_token_count, -1
        )  # shape: (batchsize, start_token_count, embed_dim)
        first_stage_embed += self.pos_emb_start_token.expand(
            batch_size, -1, -1
        )  # add positional embedding (batch_size, start_token_count, embed_dim)

        # Embed tokens from teacher forcing
        embedded_tokens = self.token_embedding(
            teacher_forcing_tokens.float()
        )  # shape: (batch_size, total_tokens-start_token_count, embed_dim)

        embedded_tokens = embedded_tokens  # shape: (batch_size, total_tokens, embed_dim)

        # Concatenate first-stage embeddings with teacher-forced tokens
        hidden_states = torch.cat(
            (first_stage_embed, embedded_tokens), dim=1
        )  # shape: (batch_size, total_tokens, embed_dim)

        # Add positional and stage-level embeddings
        hidden_states = (
            hidden_states + self.pos_emb_all_stages[:, : self.total_tokens]
        )  # shape: (1, total_tokens, embed_dim) each position of each token mapped to its embedding
        +self.stage_level_embedding(
            self.stage_indices[:, : self.total_tokens]
        )  # shape: (1, total_tokens, embed_dim) each stage of each token mapped to its embedding)

        # Retrieve causal attention bias
        attention_bias = self.causal_attn_bias[:, :, : self.total_tokens, : self.total_tokens]
        # Process hidden states through each Transformer block
        for block in self.blocks:
            hidden_states = block(
                x=hidden_states,
                conditioning_tensor=condition_embeddings,
                attention_bias=attention_bias,
            )

        # Compute logits from final hidden states for ce loss
        logits = self.compute_logits(hidden_states.float(), condition_embeddings)

        return logits

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        batch_size: int,
        class_labels: Optional[Union[int, torch.LongTensor]],
        seed: Optional[int] = None,
        guidance_scale: float = 1.5,
        top_k: int = 0,
        top_p: float = 0.0,
        use_softmax_smoothing: bool = False,
    ) -> torch.Tensor:
        """
        Perform autoregressive inference with classifier-free guidance (CFG).

        Args:
            batch_size (int): Batch size.
            class_labels (int or torch.LongTensor, optional): Class label(s).
                If None, label(s) will be sampled randomly.
                If an integer, will be broadcast to the entire batch.
                If < 0, uses 'num_classes' index to represent a dropped label (CFG).
            seed (int, optional): Random seed for reproducible sampling.
            guidance_scale (float): Classifier-free guidance scale.
            top_k (int): Top-k filtering for sampling.
            top_p (float): Top-p (nucleus) filtering for sampling.
            use_softmax_smoothing (bool): If True, apply a Gumbel-softmax “smoothing”
                step (used mainly for visualization, not for evaluation metrics).

        Returns:
            torch.Tensor: A reconstructed image of shape (B, 3, H, W) in [0, 1].
        """

        # 1. Handle random seed
        if seed is None:
            random_generator = None
        else:
            self.random_generator.manual_seed(seed)
            random_generator = self.random_generator

        # 2. Handle class labels
        if class_labels is None:
            # Sample labels from uniform distribution
            class_labels = torch.multinomial(
                self.uniform_class_prob,
                num_samples=batch_size,
                replacement=True,
                generator=random_generator,
            ).reshape(batch_size)
        elif isinstance(class_labels, int):
            # If label < 0, use self.num_classes as the "dropped" label index
            fill_value = self.num_classes if class_labels < 0 else class_labels
            class_labels = torch.full(
                (batch_size,),
                fill_value=fill_value,
                device=self.stage_indices.device,
            )

        # print("debug1")
        # 3. Create the "start of sequence" conditioning by concatenating:
        #   [real_label, dropped_label_placeholder]
        # and embedding them. This doubles the batch dimension (2*B) for CFG.

        class_labels_extended = torch.cat(
            (
                class_labels,
                torch.full_like(class_labels, fill_value=self.num_classes),
            ),
            dim=0,
        )

        conditioning_embeddings = self.class_embedding(class_labels_extended)  # shape: (2*B, embed_dim)

        # conditioning_embeddings = torch.randn(2 * batch_size, 128).to("cuda")
        start_conditioning = conditioning_embeddings[:batch_size]  # shape: (B, embed_dim)

        # print("debug2")
        # 4. Sum level + position embeddings for all tokens
        #    (Example naming; adjust if your code uses them differently.)
        stage_pos_encoding = self.stage_level_embedding(self.stage_indices) + self.pos_emb_all_stages

        # 5. Construct the initial hidden states for the first stage
        #    We add:
        #       * start_conditioning (2*B, start_token_count, embed_dim)
        #       * a special "start" positional embedding
        #       * the stage_pos_encoding slice for the first stage
        next_hidden_states = (
            start_conditioning.unsqueeze(1).expand(2 * batch_size, self.start_token_count, -1)  # (2*B, 1, embed_dim)
            + self.pos_emb_start_token.expand(2 * batch_size, self.start_token_count, -1)
            + stage_pos_encoding[:, : self.start_token_count]
        )

        # print("debug3")
        # Prepare a latent canvas to fill in with the quantized tokens
        reconstructed_latent = start_conditioning.new_zeros(
            batch_size,
            self.codebook_vectors_dim,
            self.resolutions_list[-1],
            self.resolutions_list[-1],
        )

        # Keep track of how many tokens we've “consumed” so far
        consumed_token_count = 0

        # print("debug4")

        # 6. Loop over each stage
        for b in self.blocks:
            b.attention.kv_caching(True)
        for stage_idx, patch_size in enumerate(self.resolutions_list):
            progress_ratio = stage_idx / (self.scale_count - 1)
            consumed_token_count += patch_size * patch_size
            # print("debug4.1")

            # Transformer forward pass
            hidden_states = next_hidden_states
            for block in self.blocks:
                # print("debug4.2")
                hidden_states = block(
                    x=hidden_states,
                    conditioning_tensor=conditioning_embeddings,
                    attention_bias=None,
                )

            logits = self.compute_logits(hidden_states, conditioning_embeddings)

            # print("debug5")

            # 7. Classifier-Free Guidance interpolation
            #    Split the logits along the batch dimension:
            #       real-branch = logits[:B]
            #       dropped-branch = logits[B:]
            t = guidance_scale * progress_ratio
            logits = (1.0 + t) * logits[:batch_size] - t * logits[batch_size:]

            # 8. Sample from logits (top-k / top-p)
            sampled_indices = sample_with_top_k_and_top_p(
                logits,
                generator=random_generator,
                top_k=top_k,
                top_p=top_p,
                num_samples=1,
            )[
                :, :, 0
            ]  # shape: (B, num_tokens)

            # print("debug6")

            # 9. Convert sampled token indices → VAE-quantized embeddings
            if not use_softmax_smoothing:
                # Standard discrete sampling
                quantized_tokens = self.vqvae_quantizer_reference.embedding(sampled_indices)
                # shape: (B, num_tokens, codebook_vectors_dim)
            else:
                # Gumbel Softmax smoothing (for visualization)
                gum_t = max(0.27 * (1 - progress_ratio * 0.95), 0.005)
                smooth_weights = gumbel_softmax_with_generator(
                    logits.mul(1.0 + progress_ratio),
                    tau=gum_t,
                    hard=False,
                    dim=-1,
                    rng=random_generator,
                )
                # Weighted sum of codebook vectors
                quantized_tokens = smooth_weights @ self.vqvae_quantizer_reference.embedding.weight.unsqueeze(0)
                # shape: (B, num_tokens, codebook_vectors_dim)

            # Move the channels dimension back to (B, codebook_vectors_dim, patch_size, patch_size)
            quantized_tokens = quantized_tokens.transpose(1, 2).reshape(
                batch_size, self.codebook_vectors_dim, patch_size, patch_size
            )

            # 10. Update the “reconstructed_latent” and prepare the next hidden states
            reconstructed_latent, next_hidden_states = self.vqvae_quantizer_reference.get_next_autoregressive_input(
                stage_idx,
                len(self.resolutions_list),
                reconstructed_latent,
                quantized_tokens,
            )

            # If not at the final stage, embed next_hidden_states for the next pass
            if stage_idx != (self.scale_count - 1):
                # shape: (B, codebook_vectors_dim, patch_size^2) → (B, patch_size^2, codebook_vectors_dim)
                next_hidden_states = next_hidden_states.view(batch_size, self.codebook_vectors_dim, -1).transpose(1, 2)
                # token_embedding + positional embed
                next_hidden_states = (
                    self.token_embedding(next_hidden_states)
                    + stage_pos_encoding[
                        :,
                        consumed_token_count : consumed_token_count + self.resolutions_list[stage_idx + 1] ** 2,
                    ]
                )
                # Repeat for real + dropped branches (2*B)
                next_hidden_states = next_hidden_states.repeat(2, 1, 1)

        # 11. Convert the final latent to an image in [0, 1]
        #     (Assumes your VAE decoder outputs [-1, 1], then shift/scale to [0, 1])
        for b in self.blocks:
            b.attention.kv_caching(False)
        return self.vqvae_reference.quantized_feats_to_image(reconstructed_latent).add_(1).mul_(0.5)


def main():
    # 1. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    # 2. MNIST data loading
    transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # replicate single channel to 3
        ]
    )

    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 3. Load your pretrained VQVAE
    #    Make sure this path points to your .pth file
    pretrained_vqvae = VQVAE(
        freeze_model=True,
        num_codebook_vectors=2048,
        latent_channels=16,
        base_channels=32,
        resolutions_list=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16),
    ).to(device)

    ckpt_path = "vqvae_results/vqvae_mnist.pth"  # Replace with your actual checkpoint
    pretrained_vqvae.load_state_dict(torch.load(ckpt_path, map_location=device))
    pretrained_vqvae.eval()

    # print("Pretrained VQVAE loaded successfully.")

    # 4. Instantiate VAR, referencing the same quantizer as the pretrained VQVAE
    var_model = VAR(
        vqvae=pretrained_vqvae,
        num_classes=10,  # Adjust if needed
        depth=6,
        embed_dim=128,
        num_heads=8,
        mlp_ratio=2.0,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        dropout_path_rate=0.01,
        cond_dropout_rate=0.01,
        norm_eps=1e-6,
        resolutions_list=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16),
    ).to(device)

    ckpt_path = "var_mnist_10.pth"  # Replace with your actual checkpoint
    var_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    var_model.eval()
    # print("VAR model instantiated.")

    # 5. Get a single batch from the test set
    example_images, example_labels = next(iter(test_loader))
    example_images = example_images.to(device)
    example_labels = example_labels.to(device)
    # print(f"Example batch shape: {example_images.shape} | labels: {example_labels}")

    # 6. We want teacher forcing tokens of shape:
    #    (B, total_tokens - start_token_count, codebook_dim)
    total_tokens = sum(ps**2 for ps in var_model.resolutions_list)
    start_token_count = var_model.resolutions_list[0] ** 2
    teacher_forcing_count = total_tokens - start_token_count
    codebook_dim = var_model.vqvae_quantizer_reference.codebook_vectors_dim

    # For a real application, you'd encode your images into codebook embeddings/indices.
    # For now, we just create random teacher-forcing tokens as a placeholder:
    batch_size = example_images.size(0)
    teacher_forcing_tokens = torch.randn(
        batch_size,
        teacher_forcing_count,
        codebook_dim,
        device=device,
    )

    # 7. Forward pass through VAR
    with torch.no_grad():
        logits = var_model(example_labels, teacher_forcing_tokens)

    # print(f"VAR output logits shape: {logits.shape}")
    # Should be: (batch_size, total_tokens, num_codebook_vectors)

    print(
        "Logits summary:",
        f"min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}",
    )

    # --- (b) Force a specific class label (e.g., digit '2')
    #     We'll generate the digit '2' for all images in the batch.
    #     If you want 'dropped' label, pass a negative int (e.g., -1).
    for i in range(10):
        forced_label = i
        samples_forced_label = var_model.autoregressive_infer_cfg(
            batch_size=batch_size,
            class_labels=forced_label,
            seed=None,
            guidance_scale=2,
            top_k=0,
            top_p=0.0,
            use_softmax_smoothing=False,
        )
        print(f"[Forced Label = {forced_label}] Generated samples shape: {samples_forced_label.shape}")

        # Save for inspection
        vutils.save_image(
            samples_forced_label.cpu(),
            f"samples_label_{forced_label}.png",
            nrow=4,
            normalize=False,
        )

    # If you want to try top-k or top-p:
    # For example, top_k=50, top_p=0.9, etc.

    print("All sample images have been saved.")


if __name__ == "__main__":
    main()
