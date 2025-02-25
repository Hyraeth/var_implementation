################################################################################
# Description: This file contains the implementation of the MultiScaleVectorQuantizer
# class, which is a multi-scale vector quantization module that discretizes features
# at progressively larger patch sizes, accumulating a residual at each stage. This
# approach is more flexible than a single-scale VQ. The class also contains helper
# classes for the Convolutional Refinement ("Phi_k") Layers.
################################################################################

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Optional, Sequence, Tuple, Union


################################################################################
# Helper Classes for the Convolutional Refinement ("Phi") Layers
################################################################################


class Phi(nn.Conv2d):
    """
    A small 3×3 convolution layer that partially refines the embedding code.

    The forward pass blends:
        output = (1 - resi_ratio) * input + (resi_ratio) * Conv(input)
    """

    def __init__(self, embed_dim: int, refinement_ratio: float):
        """
        Args:
            embed_dim: Dimension of the embedding (channel size).
            refinement_ratio: How strongly this conv transforms the input.
                              0 = no transform (identity), 1 = fully conv.
        """
        super().__init__(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.resi_ratio = max(0.0, min(1.0, abs(refinement_ratio)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: A [B, embed_dim, H, W] feature tensor.

        Returns:
            The refined feature tensor after partial convolution blending.
        """
        conv_out = super().forward(x)
        return x * (1 - self.resi_ratio) + conv_out * self.resi_ratio


class PhiNonShared(nn.ModuleList):
    """
    Each scale gets its own distinct Phi (no sharing). Use get_phi(scale_ratio)
    to retrieve the appropriate Phi based on progression through scales.
    """

    def __init__(self, phi_list: List[Phi]):
        """
        Args:
            phi_list: A list of separate Phi instances, one per scale.
        """
        super().__init__(phi_list)
        num_phis = len(phi_list)

        self.ticks = np.linspace(
            1 / (2 * num_phis),  # Start at middle of first interval
            1 - 1 / (2 * num_phis),  # End at middle of last interval
            num_phis,
        )

    def get_phi(self, scale_ratio: float) -> Phi:
        """
        Args:
            scale_ratio: A float in [0, 1] indicating progression through scales.

        Returns:
            The Phi instance whose tick is closest to scale_ratio.
        """
        scale_ratio = max(0.0, min(1.0, scale_ratio))
        index = np.argmin(np.abs(self.ticks - scale_ratio))
        return self[index]


################################################################################
# Main Class: MultiScaleVectorQuantizer
################################################################################


class MultiScaleVectorQuantizer(nn.Module):
    """
    A multi-scale vector quantization module that discretizes features at
    progressively larger patch sizes, accumulating a residual at each stage.

    This approach is more flexible than a single-scale VQ:
      1) We begin at a small patch size to capture coarse/global features.
      2) We upsample the chosen code embeddings to the full size (or next scale).
      3) We subtract this partial reconstruction from the feature maps to get
         a residual, which is then quantized again at a larger patch size.

    Optionally, each scale's embedded code can pass through a small 3×3 conv
    refinement layer (`Phi`).
    """

    def __init__(
        self,
        num_codebook_vectors: int,
        codebook_vectors_dim: int,
        commitment: float = 0.25,
        resolutions_list: Optional[Sequence[int]] = None,
        quant_residual_ratio: float = 0.5,
    ):
        """
        Args:
            num_codebook_vectors: Number of codes in the embedding dictionary.
            codebook_vectors_dim:  Dimensionality of each embedding vector.
            commitment : Weight for the commitment loss term in VQ.
            resolutions_list:
                A list of patch sizes from smaller to larger
                (e.g., [8, 16, 32]) for multi-scale quantization.
            quant_residual_ratio:
                The strength used by the Phi refinement modules
                (a ratio in [0, 1]).
        """
        super().__init__()

        self.num_codebook_vectors: int = num_codebook_vectors
        self.codebook_vectors_dim: int = codebook_vectors_dim
        self.resolutions_list: Tuple[int] = tuple(resolutions_list or [])
        self.commitment: float = commitment

        # A small convolutional refinement ratio (Phi layers).
        self.quant_residual_ratio = quant_residual_ratio

        # one Phi per scale
        modules = []
        for _ in range(len(self.resolutions_list)):
            if abs(quant_residual_ratio) > 1e-6:
                modules.append(Phi(codebook_vectors_dim, quant_residual_ratio))
            else:
                modules.append(nn.Identity())
        self.phi_non_shared = PhiNonShared(modules)

        # Embedding dictionary for the codebook
        self.embedding = nn.Embedding(self.num_codebook_vectors, self.codebook_vectors_dim)

        # Buffer for usage stats across scales (used in forward pass)
        self.register_buffer(
            "ema_vocab_hit_SV",
            torch.full((len(self.resolutions_list), self.num_codebook_vectors), 0.0),
        )
        self.record_hit = 0  # internal usage for exponential moving average usage

    def eini(self, init_std: float):
        """
        Initialize the embedding vectors in codebook.

        Args:
            init_std: If > 0, we do a truncated normal with std=init_std.
                      If < 0, we do uniform in [-abs(init_std)/num_codebook_vectors, +abs(init_std)/num_codebook_vectors].
        """
        if init_std > 0:
            nn.init.trunc_normal_(self.embedding.weight.data, std=init_std)
        elif init_std < 0:
            val = abs(init_std) / self.num_codebook_vectors
            self.embedding.weight.data.uniform_(-val, val)

    ############################################################################
    # Forward (training) - multi-scale quantization + codebook loss
    ############################################################################
    def forward(
        self, in_features: torch.Tensor, return_usage_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[float]], torch.Tensor]:
        """
        The main forward pass for training time.

        1) Initialize the residual as the input features.
        2) For each scale in resolutions_list (from small to large):
           - Downsample 'residual_features' if needed. (interpolate(f,h_k,w_k))
           - Find the closest embedding index for each spatial location. (Quantize)
           - Upsample the chosen embedding back to (H, W) and run through a Phi module. (interpolate(f,h_K,w_K) -> LookUp -> Phi)
           - Add it to 'reconstructed_features' and subtract it from 'residual_features'.
        3) Compute the multi-scale VQ loss (a combination of codebook and commitment loss).

        Args:
            in_features: Float Tensor of shape (B, C, H, W).
            return_usage_stats: If True, also return a list with usage % per scale.

        Returns:
            A tuple of:
                - final_reconstruction: The final multi-scales features
                - usage_percentages: A list of code usage percentages (one per scale) if requested
                - mean_vq_loss: The scalar VQ loss averaged over scales
        """
        if in_features.dtype != torch.float32:
            in_features = in_features.float()

        # Initialize residual and reconstructed features
        B, C, H, W = in_features.shape
        f_no_grad = in_features.detach()
        residual_features = f_no_grad.clone()
        reconstructed_features = torch.zeros_like(residual_features)

        # We'll accumulate code usage and the VQ loss across scales.
        total_vq_loss = torch.tensor(0.0, device=in_features.device)

        scales_count = len(self.resolutions_list)

        patch_hws = [(p, p) if isinstance(p, int) else (p[0], p[1]) for p in self.resolutions_list]

        # The last resolution must match the input resolution
        assert (
            patch_hws[-1][0] == H and patch_hws[-1][1] == W
        ), f"Last resolution {patch_hws[-1]} must match input (H,W)=({H},{W})"

        # Loop over each scale from the smallest patch size to the largest
        for scale_idx, resolution in enumerate(self.resolutions_list):
            # 1) Downsample or not
            if scale_idx != scales_count - 1:
                # Downsample to resolution x resolution
                curr_features = (
                    F.interpolate(residual_features, size=(resolution, resolution), mode="area")
                    .permute(0, 2, 3, 1)
                    .reshape(-1, C)
                )
            else:
                # Final scale uses the original (H, W) shape
                curr_features = residual_features.permute(0, 2, 3, 1).reshape(-1, C)

            # 2) Codebook lookup: either by  L2 distance
            # dist(a, b)^2 = |a|^2 + |b|^2 - 2 a·b
            # reduce computation by skipping square root
            dist_sq = torch.sum(curr_features.square(), dim=1, keepdim=True) + torch.sum(
                self.embedding.weight.data.square(), dim=1
            )
            dist_sq.addmm_(curr_features, self.embedding.weight.data.T, alpha=-2, beta=1)
            indices_min = torch.argmin(dist_sq, dim=1)

            # 3) Record code usage
            hit_counts = indices_min.bincount(minlength=self.num_codebook_vectors).float()

            # Reshape for reconstruction
            idx_map_BHW = indices_min.view(B, resolution, resolution)
            # 4) Retrieve embeddings and upsample them to (H, W) if needed
            embed_BCHW = self.embedding(idx_map_BHW).permute(0, 3, 1, 2).contiguous()
            if scale_idx != scales_count - 1:
                embed_BCHW = F.interpolate(embed_BCHW, size=(H, W), mode="bicubic")

            # 5) Refine with a Phi layer
            scale_ratio = scale_idx / (scales_count - 1) if (scales_count - 1) > 0 else 1.0

            embed_BCHW = self.phi_non_shared.get_phi(scale_ratio)(embed_BCHW)

            # 6) Update reconstructed_features and residual
            reconstructed_features = reconstructed_features + embed_BCHW
            residual_features = residual_features - embed_BCHW

            # Update code usage stats
            if self.record_hit == 0:
                self.ema_vocab_hit_SV[scale_idx].copy_(hit_counts)
            elif self.record_hit < 100:
                self.ema_vocab_hit_SV[scale_idx].mul_(0.9).add_(hit_counts.mul(0.1))
            else:
                self.ema_vocab_hit_SV[scale_idx].mul_(0.99).add_(hit_counts.mul(0.01))
            self.record_hit += 1

            # 7) Compute partial VQ loss at this scale
            #    We combine two losses:
            #       - MSE(f_hat, f_no_grad) to push the reconstruction close to f_no_grad
            #       - MSE(f_hat.data, f) times commitment  (commitment/codebook)
            scale_vq_loss = (
                F.mse_loss(reconstructed_features, f_no_grad)
                + F.mse_loss(reconstructed_features.data, in_features) * self.commitment
            )
            total_vq_loss = total_vq_loss + scale_vq_loss

        # Average the total VQ loss over the number of scales
        total_vq_loss /= float(scales_count)

        # Final reconstruction is a sum of the embedded codes.
        # (We remove the gradient from the difference but keep the final.)
        reconstructed_features = (reconstructed_features.data - f_no_grad) + in_features

        usage_percentages = None
        if return_usage_stats:
            # margin is a rough threshold for code usage to count as "active"
            margin = 1 * (in_features.numel() / in_features.shape[1]) / self.num_codebook_vectors * 0.08
            usage_percentages = []
            for scale_idx, resolution in enumerate(self.resolutions_list):
                hits = self.ema_vocab_hit_SV[scale_idx]
                usage_pct = (hits >= margin).float().mean().item() * 100.0
                usage_percentages.append(usage_pct)

        return reconstructed_features, usage_percentages, total_vq_loss

    ########################################################################
    # Below are additional methods that perform or assist in multi-scale VQ
    # but are not typically used in a standard forward pass.
    ########################################################################

    def embed_to_reconstruction(
        self,
        multi_scale_embeddings: List[torch.Tensor],
        return_only_last: bool = False,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Given a list of per-scale embeddings (of shape [B, C, H_scale, W_scale]),
        reconstruct them to a common size (usually the largest scale) and optionally
        refine each one with Phi, accumulating the result.

        Args:
            multi_scale_embeddings (List[torch.Tensor]):
                Embeddings at each scale from smallest to largest, e.g. [B, C, 8, 8], [B, C, 16, 16], ...
            return_only_last (bool):
                If True, return only the final reconstruction; otherwise return a list
                of reconstructions after each scale.

        Returns:
            Union[List[torch.Tensor], torch.Tensor]:
                - If return_only_last = False, returns a list of shape [S] with each
                  partial reconstruction at the largest scale.
                - If return_only_last = True, returns a single [B, C, max_H, max_W] tensor.
        """
        if not multi_scale_embeddings:
            return []

        scales_count = len(multi_scale_embeddings)
        batch_size = multi_scale_embeddings[0].shape[0]
        max_height = max_width = self.resolutions_list[-1]

        reconstructed_list = []
        # Start accumulation with a zero tensor at the final scale's size
        reconstructed_feature = multi_scale_embeddings[-1].new_zeros(
            batch_size,
            self.codebook_vectors_dim,
            max_height,
            max_width,
            dtype=torch.float32,
        )

        for scale_idx, resolution in enumerate(self.resolutions_list):
            embed_BCHW = multi_scale_embeddings[scale_idx]
            if scale_idx < scales_count - 1:
                # Upsample to the largest scale if not already at the last
                embed_BCHW = F.interpolate(embed_BCHW, size=(max_height, max_width), mode="bicubic")
            # Refine with Phi
            scale_ratio = scale_idx / (scales_count - 1) if (scales_count > 1) else 1.0
            embed_BCHW = self.phi_non_shared.get_phi(scale_ratio)(embed_BCHW)

            reconstructed_feature.add_(embed_BCHW)
            if return_only_last:
                reconstructed_list = reconstructed_feature
            else:
                reconstructed_list.append(reconstructed_feature.clone())

        return reconstructed_list

    def quantize_features_multiscale(
        self,
        input_features: torch.Tensor,
        return_reconstruction: bool,
        resolutions: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
    ) -> List[Union[torch.Tensor, torch.LongTensor]]:
        """
        For each scale in resolutions, quantize the features (either by returning
        codebook indices or by returning the partial reconstruction). The input
        is refined scale by scale, subtracting out the partial reconstruction
        as we go.

        Args:
            input_features (torch.Tensor): [B, C, H, W] float tensor.
            return_reconstruction (bool):
                If True, return partial reconstructions at each scale.
                If False, return the codebook indices (flattened) at each scale.
            resolutions (List[int or (int,int)], optional):
                resolution of token maps from small to large. If None, uses self.resolutions_list.

        Returns:
            List[torch.Tensor or torch.LongTensor]:
                A list where each element is either:
                    - partial reconstruction at that scale, or
                    - the flattened codebook indices at that scale.
        """
        B, C, H, W = input_features.shape
        # This is the "residual_features" we keep subtracting from
        residual_features = input_features.detach().clone()
        # We'll accumulate partial reconstructions in `reconstructed_features`
        reconstructed_features = torch.zeros_like(residual_features)

        resolutions = resolutions or self.resolutions_list
        patch_hws = [(p, p) if isinstance(p, int) else (p[0], p[1]) for p in resolutions]

        # The last patch size must match the input size
        assert (
            patch_hws[-1][0] == H and patch_hws[-1][1] == W
        ), f"Last patch size {patch_hws[-1]} must match input (H,W)=({H},{W})"

        results = []
        scales_count = len(patch_hws)

        for scale_idx, (ph, pw) in enumerate(patch_hws):
            curr_features = residual_features

            if scale_idx != scales_count - 1:
                # Downsample to resolution x resolution
                curr_features = F.interpolate(curr_features, size=(ph, pw), mode="area")

            curr_features = curr_features.permute(0, 2, 3, 1).reshape(-1, C)

            # Codebook lookup: either by L2 distance
            # dist(a, b)^2 = |a|^2 + |b|^2 - 2 a·b
            # reduce computation by skipping square root
            dist_sq = torch.sum(curr_features.square(), dim=1, keepdim=True) + torch.sum(
                self.embedding.weight.data.square(), dim=1
            )
            dist_sq.addmm_(curr_features, self.embedding.weight.data.T, alpha=-2, beta=1)
            indices_min = torch.argmin(dist_sq, dim=1)

            # Reshape indices to B,ph,pw
            idx_map_BHW = indices_min.view(B, ph, pw)
            # Retrieve embeddings and possibly upsample
            embed_BCHW = self.embedding(idx_map_BHW).permute(0, 3, 1, 2).contiguous()  # (B, codebook_vec_dim, ph, pw)

            if scale_idx != scales_count - 1:
                embed_BCHW = F.interpolate(embed_BCHW, size=(H, W), mode="bicubic")

            # Refine with Phi
            scale_ratio = scale_idx / (scales_count - 1) if (scales_count - 1) > 0 else 1.0

            embed_BCHW = self.phi_non_shared.get_phi(scale_ratio)(embed_BCHW)

            # Update reconstructed_features and residual_features
            reconstructed_features.add_(embed_BCHW)
            residual_features.sub_(embed_BCHW)

            if return_reconstruction:
                results.append(reconstructed_features.clone())
            else:
                # Return codebook indices flattened per scale
                results.append(indices_min.view(B, ph * pw))

        return results

    def indices_to_var_input(self, gt_scale_indices: List[torch.Tensor]) -> torch.Tensor:
        """
        Convert a list of ground-truth codebook indices at multiple scales
        into a single input for next stage AR model.
        This is used in teacher-forcing setups.

        Args:
            gt_scale_indices (List[torch.Tensor]):
                A list of index tensors at each scale (not necessarily the same resolution).
                Typically shape: [B, ph*pw] flattened indices for scale.

        Returns:
            torch.Tensor: A float32 tensor of shape [B, L, C] (where L is the total
                          number of tokens across scales) used for next-stage input.
        """
        if not gt_scale_indices:
            return None

        batch_size = gt_scale_indices[0].shape[0]
        channels = self.codebook_vectors_dim
        final_size = self.resolutions_list[-1]  # Must match final (H, W)

        reconstructed_features = gt_scale_indices[0].new_zeros(
            batch_size,
            channels,
            final_size,
            final_size,
            dtype=torch.float32,
        )
        inputs_for_next_scales = []

        codebook_indices = len(gt_scale_indices)
        current_resolution = self.resolutions_list[0]

        for scale_idx in range(codebook_indices - 1):
            # Indices at this scale
            idx_map_bhw = gt_scale_indices[scale_idx].view(batch_size, current_resolution, current_resolution)
            embed_BCHW = self.embedding(idx_map_bhw)

            # Upsample to final size, refine, accumulate
            embed_BCHW = F.interpolate(
                embed_BCHW.permute(0, 3, 1, 2),
                size=(final_size, final_size),
                mode="bicubic",
            )
            scale_ratio = scale_idx / (codebook_indices - 1)
            refined = self.phi_non_shared.get_phi(scale_ratio)(embed_BCHW)
            reconstructed_features.add_(refined)

            # Prepare for the next scale
            next_size = self.resolutions_list[scale_idx + 1]
            next_input = (
                F.interpolate(reconstructed_features, size=(next_size, next_size), mode="area")
                .view(batch_size, channels, -1)
                .transpose(1, 2)
            )
            inputs_for_next_scales.append(next_input)

            current_resolution = next_size

        if inputs_for_next_scales:
            return torch.cat(inputs_for_next_scales, dim=1)
        else:
            return None

    def get_next_autoregressive_input(
        self,
        scale_idx: int,
        scales_count: int,
        accumulated_recon: torch.Tensor,
        new_embeddings_bchw: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Used in an autoregressive (AR) setting: given the current partial reconstruction
        and newly generated embeddings at this scale, refine, upsample (if needed),
        and produce the next input.

        Args:
            scale_idx (int): Which scale index we are currently processing.
            scales_count (int): Total number of scales.
            accumulated_recon (torch.Tensor): Current partial reconstruction of size [B, C, H, W].
            new_embeddings_bchw (torch.Tensor): Embeddings at this scale's resolution [B, C, h, w].

        Returns:
            (Optional[torch.Tensor], torch.Tensor):
                - If this is not the last scale, returns (updated_reconstructed_features, next_downsampled_input)
                - If this is the last scale, returns (final_reconstruction, final_reconstruction).
        """
        final_size = self.resolutions_list[-1]
        scale_ratio = (scale_idx / (scales_count - 1)) if (scales_count > 1) else 1.0

        if scale_idx < scales_count - 1:
            # Upsample to final size, refine, update
            refined = self.phi_non_shared.get_phi(scale_ratio)(
                F.interpolate(new_embeddings_bchw, size=(final_size, final_size), mode="bicubic")
            )
            accumulated_recon.add_(refined)
            # Prepare next input
            next_size = self.resolutions_list[scale_idx + 1]
            next_input = F.interpolate(accumulated_recon, size=(next_size, next_size), mode="area")
            return accumulated_recon, next_input
        else:
            # Last scale, no more upsampling needed
            refined = self.phi_non_shared.get_phi(scale_ratio)(new_embeddings_bchw)
            accumulated_recon.add_(refined)
            return accumulated_recon, accumulated_recon


################################################################################
# Testing the MultiScaleVectorQuantizer
################################################################################

if __name__ == "__main__":
    # Test the MultiScaleVectorQuantizer
    # Create a dummy MultiScaleVectorQuantizer
    ms_vq = MultiScaleVectorQuantizer(
        num_codebook_vectors=4096,
        codebook_vectors_dim=128,
        commitment=0.25,
        resolutions_list=[8, 16, 32, 64],
        quant_residual_ratio=0.5,
    )

    # Initialize the embedding vectors
    ms_vq.eini(0.02)

    # Create a dummy input tensor
    input_tensor = torch.randn(3, 128, 64, 64)
    print("number of Codebook Vectors:", ms_vq.num_codebook_vectors)
    print("Codebook Vectors Dim:", ms_vq.codebook_vectors_dim)

    # Forward pass
    reconstructed_features, usage_percentages, total_vq_loss = ms_vq(input_tensor, True)

    print("Reconstructed Features Shape:", reconstructed_features.shape)
    print("Usage Percentages:", usage_percentages)
    print("Total VQ Loss:", total_vq_loss.item())
    print("Forward Pass Test Passed!\n")

    # Test the quantize_features_multiscale method
    results = ms_vq.quantize_features_multiscale(input_tensor, return_reconstruction=True)
    print("Results Length:", len(results))
    for i, result in enumerate(results):
        print(f"Result {i} Shape:", result.shape)
    print("Quantize Features Multiscale Test Passed!\n")

    # Test the embed_to_reconstruction method
    reconstructed_list = ms_vq.embed_to_reconstruction(results)
    print("Reconstructed List Shape:", reconstructed_list[0].shape)
    print("Embed to Reconstruction Test Passed!\n")

    # Test the indices_to_var_input method
    indices = ms_vq.quantize_features_multiscale(input_tensor, return_reconstruction=False)
    print("Indices Length:", len(indices))
    for i, index in enumerate(indices):
        print(f"Index {i} Shape:", index.shape)
    next_input = ms_vq.indices_to_var_input(indices)
    print("Next Input Shape:", next_input.shape)
    print("Indices to Var Input Test Passed!\n")

    # Test the get_next_autoregressive_input method
    next_input, next_input = ms_vq.get_next_autoregressive_input(1, 3, reconstructed_features, results[1])
    print("Next Input Shape:", next_input.shape)
    print("Get Next Autoregressive Input Test Passed!\n")

    print("All Tests Passed!")
