import torch
import torch.nn as nn
from typing import List, Optional, Sequence, Tuple, Union

from vae import Encoder, Decoder
from multiscale_vec_quant import (
    MultiScaleVectorQuantizer,
)

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import torch.optim as optim
import matplotlib.pyplot as plt
import unittest


class VQVAE(nn.Module):
    """
    A multi-scale VQ-VAE model that uses an encoder, decoder, and a vector quantizer.

    Args:
        num_codebook_vectors (int): Size of the discrete codebook (number of embeddings).
        latent_channels (int): Dimensionality (channel count) of the latent space.
        base_channels (int): Base number of feature maps for the encoder/decoder.
        dropout_rate (float): Dropout rate used in ResNet blocks.
        commitment_weight (float): The beta (commitment) weight for the VQ loss.
        quant_conv_kernel_size (int): Kernel size for the pre/post-conv layers around the quantizer.
        quant_residual_ratio (float): Ratio for blending identity + conv in the residual refiner (φ).
        resolutions_list (Sequence[int]): A list of patch resolutions for multi-scale quantization.
        freeze_model (bool): If True, switch to eval mode and disable parameter gradients (like a frozen model).

    Note:
        - The `Encoder` and `Decoder` are imported from basic_vae.py, which follow the new naming:
          Encoder(base_channels, channel_multipliers, ...) and
          Decoder(base_channels, channel_multipliers, ...).
        - The MultiScaleVectorQuantizer can be replaced by your new MultiScaleVectorQuantizer if desired.
    """

    def __init__(
        self,
        *,
        num_codebook_vectors: int = 2048,
        latent_channels: int = 32,
        base_channels: int = 128,
        dropout_rate: float = 0.0,
        commitment_weight: float = 0.25,  # also called "beta" in some references
        quant_conv_kernel_size: int = 3,
        quant_residual_ratio: float = 0.5,
        resolutions_list: Sequence[int] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16),
        freeze_model: bool = True,
    ):
        super().__init__()

        # Store constructor parameters
        self.num_codebook_vectors = num_codebook_vectors
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.dropout_rate = dropout_rate
        self.commitment_weight = commitment_weight
        self.quant_conv_kernel_size = quant_conv_kernel_size
        self.quant_residual_ratio = quant_residual_ratio
        self.resolutions_list = resolutions_list
        self.freeze_model = freeze_model

        encoder_decoder_config = dict(
            base_channels=self.base_channels,
            channel_multipliers=(1, 2, 4),  # example setting; adapt as needed
            num_resnet_blocks=2,
            dropout_rate=self.dropout_rate,
            in_channels=3,  # For RGB input
            latent_channels=self.latent_channels,
        )

        # Instantiate the encoder and decoder
        self.encoder = Encoder(**encoder_decoder_config)
        self.decoder = Decoder(**encoder_decoder_config)

        # Downsample ratio from the sum of channel multipliers
        # or specifically 2^(len(channel_multipliers) - 1), etc.
        self.downsample_factor = 2 ** (len(encoder_decoder_config["channel_multipliers"]) - 1)

        print("downsample_factor", self.downsample_factor)

        # Vector quantizer (multi-scale)
        self.vector_quantizer = MultiScaleVectorQuantizer(
            num_codebook_vectors=self.num_codebook_vectors,
            codebook_vectors_dim=self.latent_channels,
            commitment=self.commitment_weight,
            resolutions_list=self.resolutions_list,
            quant_residual_ratio=self.quant_residual_ratio,
        )

        # Pre-/Post-conv layers around the quantizer
        self.pre_quant_conv = nn.Conv2d(
            self.latent_channels,
            self.latent_channels,
            self.quant_conv_kernel_size,
            stride=1,
            padding=self.quant_conv_kernel_size // 2,
        )
        self.post_quant_conv = nn.Conv2d(
            self.latent_channels,
            self.latent_channels,
            self.quant_conv_kernel_size,
            stride=1,
            padding=self.quant_conv_kernel_size // 2,
        )

        # If freeze_model=True, set eval mode and turn off grad
        if self.freeze_model:
            self.eval()
            for param in self.parameters():
                param.requires_grad_(False)

    def forward(
        self,
        input_image: torch.Tensor,
        return_usage_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[float]], torch.Tensor]:
        """
        Forward pass for training:
        1) Encode the input image to latent features,
        2) Quantize them,
        3) Decode the quantized features back to an image.

        Args:
            input_image (torch.Tensor): Shape (B, 3, H, W).
            return_usage_stats (bool): If True, also return code usage stats.

        Returns:
            reconstructed (torch.Tensor): The reconstructed image (B, 3, H, W).
            usage_stats (List[float]): Optional usage stats (None if return_usage_stats=False).
            vq_loss (torch.Tensor): The VQ loss (commitment + codebook).
        """
        B, C, H, W = input_image.shape

        patch_hws = [(p, p) if isinstance(p, int) else (p[0], p[1]) for p in self.resolutions_list]

        # The last resolution must match the input resolution
        assert (
            patch_hws[-1][0] == H // self.downsample_factor and patch_hws[-1][1] == W // self.downsample_factor
        ), f"Last resolution {patch_hws[-1]} must match downsampled input (H,W)=({H// self.downsample_factor},{W// self.downsample_factor})"
        # Encode
        encoded_feats = self.encoder(input_image)
        encoded_feats = self.pre_quant_conv(encoded_feats)

        # Vector quantization (multi-scale)
        # print("encoded_feats.shape", encoded_feats.shape)
        quantized, usage_stats, vq_loss = self.vector_quantizer(encoded_feats, return_usage_stats=return_usage_stats)

        # Decode
        decoded_feats = self.post_quant_conv(quantized)
        reconstructed = self.decoder(decoded_feats)

        return reconstructed, usage_stats, vq_loss

    def quantized_feats_to_image(self, quantized_feats: torch.Tensor) -> torch.Tensor:
        """
        Convert already-quantized feature maps into an output image.

        Args:
            quantized_feats (torch.Tensor): Latent feature maps to decode (B, C, H, W).

        Returns:
            (torch.Tensor): Reconstructed image, clipped to [-1, 1].
        """
        decoded = self.decoder(self.post_quant_conv(quantized_feats))
        return decoded.clamp_(-1, 1)

    def image_to_code_indices(
        self,
        input_image: torch.Tensor,
        resolutions: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
    ) -> List[torch.LongTensor]:
        """
        Encode an image to code indices at each scale (without returning the intermediate reconstruction).

        Args:
            input_image (torch.Tensor): Shape (B, 3, H, W).
            resolutions (Sequence[int] or None): If specified, override the default self.resolutions_list.

        Returns:
            List[torch.LongTensor]: A list of codebook index tensors (one per scale).
        """
        encoded_feats = self.encoder(input_image)
        encoded_feats = self.pre_quant_conv(encoded_feats)
        return self.vector_quantizer.quantize_features_multiscale(
            input_features=encoded_feats,
            return_reconstruction=False,
            resolutions=resolutions,
        )

    def code_indices_to_image(
        self,
        multi_scale_indices: List[torch.Tensor],
        return_only_last: bool = False,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Convert multi-scale code indices to a reconstructed image.

        Args:
            multi_scale_indices (List[torch.Tensor]): A list of index tensors, one per scale.
            return_only_last (bool): If True, return only the final reconstructed image;
                                      otherwise, return a list of partial reconstructions.

        Returns:
            If return_only_last=True, returns (torch.Tensor) the final image.
            Otherwise, returns a list of (torch.Tensor) partial reconstructions.
        """
        batch_size = multi_scale_indices[0].shape[0]
        embed_list = []
        for idx_tensor in multi_scale_indices:
            spatial_len = idx_tensor.shape[1]  # flattened patch dimension
            patch_size = round(spatial_len**0.5)
            # Convert indices back to embeddings
            emb = self.vector_quantizer.embedding(idx_tensor).transpose(1, 2)
            emb = emb.view(batch_size, self.latent_channels, patch_size, patch_size)
            embed_list.append(emb)

        # Turn them into partial or final quantized feats
        fused_feats = self.vector_quantizer.embed_to_reconstruction(
            multi_scale_embeddings=embed_list,
            return_only_last=return_only_last,
        )

        if return_only_last:
            # single Tensor path
            return self.quantized_feats_to_image(fused_feats)
        else:
            # multiple partial reconstructions path
            images = [self.quantized_feats_to_image(f) for f in fused_feats]
            return images

    def image_to_partial_reconstructions(
        self,
        input_image: torch.Tensor,
        resolutions: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
        return_only_last: bool = False,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Encode an image to multi-scale partial reconstructions (or final only).

        Args:
            input_image (torch.Tensor): Shape (B, 3, H, W).
            resolutions (Sequence[int], optional): Override default resolutions.
            return_only_last (bool): If True, returns only the final reconstruction.

        Returns:
            If return_only_last=True, returns (torch.Tensor).
            Otherwise, returns a list of partial reconstructions (List[torch.Tensor]).
        """
        # Encode
        encoded_feats = self.encoder(input_image)
        encoded_feats = self.pre_quant_conv(encoded_feats)
        # Get partial f_hat from each scale
        partial_feats_list = self.vector_quantizer.quantize_features_multiscale(
            input_features=encoded_feats,
            return_reconstruction=True,
            resolutions=resolutions,
        )
        if return_only_last:
            final_feats = partial_feats_list[-1]
            return self.quantized_feats_to_image(final_feats)
        else:
            # Return all partial images
            return [self.quantized_feats_to_image(f) for f in partial_feats_list]


######################################################################
# Test the VQ-VAE model
######################################################################
def train(model):
    # Training configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    num_epochs = 5
    learning_rate = 3e-4
    save_dir = "vqvae_results"
    os.makedirs(save_dir, exist_ok=True)

    # MNIST data loading
    transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model initialization
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    recon_loss_fn = nn.MSELoss()
    torch.autograd.set_detect_anomaly(True)
    # Training loop
    # Initialize lists to store loss values
    train_recon_losses = []
    train_vq_losses = []
    test_recon_losses = []
    test_vq_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_recon_loss = 0.0
        total_train_vq_loss = 0.0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            reconstructed, _, vq_loss = model(data)
            recon_loss = recon_loss_fn(reconstructed, data)
            loss = recon_loss + vq_loss

            loss.backward()
            optimizer.step()

            total_train_recon_loss += recon_loss.item()
            total_train_vq_loss += vq_loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        avg_train_recon_loss = total_train_recon_loss / len(train_loader)
        avg_train_vq_loss = total_train_vq_loss / len(train_loader)
        train_recon_losses.append(avg_train_recon_loss)
        train_vq_losses.append(avg_train_vq_loss)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Avg Train Recon Loss: {avg_train_recon_loss:.4f}, Avg Train VQ Loss: {avg_train_vq_loss:.4f}"
        )

        # Evaluation on the test set
        model.eval()
        total_test_recon_loss = 0.0
        total_test_vq_loss = 0.0

        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                reconstructed, _, vq_loss = model(data)
                recon_loss = recon_loss_fn(reconstructed, data)

                total_test_recon_loss += recon_loss.item()
                total_test_vq_loss += vq_loss.item()

            avg_test_recon_loss = total_test_recon_loss / len(test_loader)
            avg_test_vq_loss = total_test_vq_loss / len(test_loader)
            test_recon_losses.append(avg_test_recon_loss)
            test_vq_losses.append(avg_test_vq_loss)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Avg Test Recon Loss: {avg_test_recon_loss:.4f}, Avg Test VQ Loss: {avg_test_vq_loss:.4f}"
        )

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_recon_losses, label="Train Reconstruction Loss")
    plt.plot(range(1, num_epochs + 1), test_recon_losses, label="Test Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_dir}/recon_loss_plot.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_vq_losses, label="Train VQ Loss")
    plt.plot(range(1, num_epochs + 1), test_vq_losses, label="Test VQ Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VQ Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_dir}/vq_loss_plot.png")
    plt.close()

    # Save final model
    torch.save(model.state_dict(), f"{save_dir}/vqvae_mnist.pth")


def test(model, test_loader, device):
    model.eval()
    total_recon_loss = 0.0
    total_vq_loss = 0.0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            reconstructed, _, vq_loss = model(data)

            recon_loss = nn.MSELoss()(reconstructed, data)
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()

    avg_recon = total_recon_loss / len(test_loader)
    avg_vq = total_vq_loss / len(test_loader)
    print(f"Test Results - Recon Loss: {avg_recon:.4f}, VQ Loss: {avg_vq:.4f}")

    # Move to CPU and convert to numpy
    data = data.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()

    # Plot original and reconstructed images
    num_images = 8  # Number of images to display
    fig, axes = plt.subplots(2, num_images, figsize=(12, 4))

    for i in range(num_images):
        # Original image
        axes[0, i].imshow(data[i].transpose(1, 2, 0))  # Assuming (C, H, W) format
        axes[0, i].axis("off")

        # Reconstructed image
        axes[1, i].imshow(reconstructed[i].transpose(1, 2, 0))
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original Images")
    axes[1, 0].set_title("Reconstructed Images")
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MNIST data loading
    transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )

    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    model = VQVAE(
        freeze_model=False,
        num_codebook_vectors=2048,
        latent_channels=16,
        base_channels=32,
        resolutions_list=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16),
    ).to(device)

    train(model)

    model.freeze_model = True
    model.load_state_dict(torch.load("vqvae_results/vqvae_mnist.pth"))

    test(model, test_loader, device)
