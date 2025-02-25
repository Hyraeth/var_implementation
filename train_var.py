import torch
from torch import nn as nn
from torch.nn import functional as F
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


from var_block import AdaptiveLayerNormBeforeHead, AdaptiveLayerNormSelfAttention
from multiscale_vqvae import VQVAE
from var import VAR

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.utils as vutils


def train_var_on_mnist(
    var_model: VAR,
    vqvae: VQVAE,
    epochs: int = 15,
    batch_size: int = 48,
    lr: float = 1e-4,
    save_path: str = "var_mnist_10.pth",
):
    """
    Train the VAR model on MNIST using a pretrained (frozen) VQVAE for codebook indices.

    Args:
        var_model (VAR): The instantiated VAR model.
        vqvae (VQVAE): Pretrained VQVAE model (used to get codebook indices).
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        save_path (str): Where to save the trained VAR checkpoint.
    """
    device = next(var_model.parameters()).device

    # 1. MNIST Datasets & Loaders
    transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # replicate 1 channel -> 3
        ]
    )

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 2. Create optimizer & loss function
    optimizer = optim.Adam(var_model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    # Freeze VQVAE so we don’t accidentally update it
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    var_model.train()  # set VAR in training mode

    # Precompute some constants for shape handling
    total_tokens = sum(ps**2 for ps in var_model.resolutions_list)
    first_stage_tokens = var_model.resolutions_list[0] ** 2  # e.g. 1^2 = 1 if first stage is size=1
    teacher_forcing_count = total_tokens - first_stage_tokens
    vocab_size = var_model.num_codebook_vectors

    loss_weight = torch.ones(1, total_tokens, device=device) / total_tokens

    print("Starting training loop...")
    for epoch in range(epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)  # (B, 3, 64, 64)
            labels = labels.to(device)  # (B,)

            # -----------------------------------------------------------
            #  Step 1: Get codebook indices from the pretrained VQVAE
            # -----------------------------------------------------------
            with torch.no_grad():
                codebook_indices = vqvae.image_to_code_indices(images)

            # Step 2 : Convert indices => embeddings (B, teacher_forcing_count, codebook_dim)
            teacher_forcing_embeds = var_model.vqvae_quantizer_reference.indices_to_var_input(codebook_indices)

            # -----------------------------------------------------------
            #  Step 3: Forward pass
            #     The VAR forward(...) returns logits of shape
            #         (B, total_tokens, vocab_size).
            # -----------------------------------------------------------
            logits = var_model(class_labels=labels, teacher_forcing_tokens=teacher_forcing_embeds)

            # -----------------------------------------------------------
            #  Step 4: Compute autoregressive cross-entropy loss
            #     We compare all predicted tokens (B, total_tokens) with
            #     the *entire* set of codebook indices (including
            #     the first-stage token).
            # -----------------------------------------------------------
            # Flatten for cross-entropy:
            #   logits => (B*total_tokens, vocab_size)
            #   targets => (B*total_tokens,)
            loss = criterion(
                logits.reshape(-1, vocab_size),
                torch.cat(codebook_indices, dim=1).view(-1),
            )

            # Backprop + step optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")

    # Save the trained VAR checkpoint
    torch.save(var_model.state_dict(), save_path)
    print(f"Finished training. Model saved to {save_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    print("Pretrained VQVAE loaded successfully.")

    var = VAR(
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

    train_var_on_mnist(var, pretrained_vqvae)
