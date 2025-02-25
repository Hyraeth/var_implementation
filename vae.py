####################################################################################################
# VAE implementation with ResNet blocks and self-attention.
# This implementation is inspired by the LDM project
# (https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py#L368).
# The Encoder and Decoder classes are implemented with ResNet blocks and self-attention modules.
# The Encoder class maps an image to a latent representation, while the Decoder class reconstructs images from latent codes.
# The Encoder class uses ResNet blocks, self-attention, and 2x downsampling steps to map an image to a latent representation.
# The Decoder class uses ResNet blocks, self-attention, and 2x upsampling steps to reconstruct images from latent codes.
####################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest


class Normalize(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.norm(x)


class ResnetBlock(nn.Module):
    """
    A standard ResNet block with GroupNorm, SILU activation, optional dropout,
    and a residual/shortcut connection.
    """

    def __init__(self, *, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convs = nn.Sequential(
            Normalize(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            Normalize(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.convs(x)


class SelfAttention(nn.Module):  # Single head self-attention
    """_
    Simple 2D self attention block.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.norm = Normalize(embed_dim)
        self.query = torch.nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        self.key = torch.nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        self.value = torch.nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        self.projection = torch.nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        # print(x.shape)

        tmp = self.norm(x)

        #  compute the Q, K, V
        q = self.query(tmp)
        k = self.key(tmp)
        v = self.value(tmp)

        #  compute the attention matrix using Q and K
        B, C, H, W = q.shape
        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # B,N,C
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, H * W)

        attention = F.softmax(torch.matmul(q, k) / (C**0.5), dim=-1)  # B,N,N

        # attend to values
        attention = attention.permute(0, 2, 1)
        output = torch.matmul(v, attention).reshape(B, C, H, W)

        output = self.projection(output)

        return x + output


def make_attention_block(in_channels, use_self_attn=True):
    return SelfAttention(in_channels) if use_self_attn else nn.Identity()


class Downsample2x(nn.Module):
    """
    A simple 2x downsampling module using a stride-2 convolution.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        # "pad=(left, right, top, bottom)"
        return self.conv(F.pad(x, pad=(0, 1, 0, 1), mode="constant", value=0))


class Encoder(nn.Module):
    """
    Encoder that maps an image (or feature map) to latent representations via
    several ResNet blocks, and 2x downsampling steps.

    The spatial resolution is reduced by 2 for each downsampling stage. At the
    end, it outputs `z_channels`

    Typical usage:
      1) Forward pass with an input of shape (B, in_channels, H, W).
      2) Receive output of shape (B, z_channels, H_out, W_out)

    Args:
        base_channels (int):
            The base number of channels. All channels scale from this value.
        channel_multipliers (Tuple[int]):
            Sequence of multipliers for channel dimensions at each resolution.
        num_resnet_blocks (int):
            Number of ResNet blocks at each resolution.
        dropout_rate (float):
            Dropout probability. If <= 1e-6, no dropout is applied.
        in_channels (int):
            Number of input image channels, e.g. 3 for RGB.
        z_channels (int):
            Number of latent channels to output
        use_self_attn (bool):
            If True, applies self-attention at the lowest (final) resolution.
        use_mid_self_attn (bool):
            If True, applies self-attention in the "middle" block.

    Shape:
        - Input: (B, in_channels, H, W)
        - Output: (B, z_channels, H_out, W_out) or (B, 2*z_channels, H_out, W_out)

      Where H_out = H / 2^(num_channel_multipliers - 1), and similarly for W_out.

    Example:
        >>> encoder = Encoder(
        ...     base_channels=128,
        ...     channel_multipliers=(1, 2, 4, 8),
        ...     num_resnet_blocks=2,
        ...     dropout_rate=0.0,
        ...     in_channels=3,
        ...     z_channels=256,
        ...     use_self_attn=True,
        ...     use_mid_self_attn=True
        ... )
        >>> x = torch.randn(4, 3, 64, 64)  # e.g. batch of 4 RGB images
        >>> out = encoder(x)
        >>> print(out.shape)
        torch.Size([4, 256, 8, 8])
    """

    def __init__(
        self,
        *,
        base_channels=128,
        channel_multipliers=(1, 2, 4, 8),
        num_resnet_blocks=2,
        dropout_rate=0.0,
        in_channels=3,
        latent_channels,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_resolutions = len(channel_multipliers)
        self.num_resnet_blocks = num_resnet_blocks
        self.in_channels = in_channels

        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()

        # in_ch_mult is used to track how channels increase step by step
        current_channels = base_channels
        for resolution_idx, mult in enumerate(channel_multipliers):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            out_channels = base_channels * mult
            # Build multiple ResNet blocks at this resolution
            for block_idx in range(num_resnet_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        dropout_rate=dropout_rate,
                    )
                )
                current_channels = out_channels

                # If we are at the bottom resolution
                if resolution_idx == self.num_resolutions - 1:
                    attn.append(SelfAttention(current_channels))
                else:
                    attn.append(nn.Identity(current_channels))

            down = nn.Module()
            down.block = block
            down.attn = attn

            # Insert downsampling layer if not at the final resolution
            if resolution_idx != self.num_resolutions - 1:
                down.downsample = Downsample2x(current_channels)

            self.down_blocks.append(down)

        # Middle block
        self.middle = nn.Module()
        self.middle.block_1 = ResnetBlock(
            in_channels=current_channels,
            out_channels=current_channels,
            dropout_rate=dropout_rate,
        )
        self.middle.attn_1 = SelfAttention(current_channels)
        self.middle.block_2 = ResnetBlock(
            in_channels=current_channels,
            out_channels=current_channels,
            dropout_rate=dropout_rate,
        )

        # Final projection
        self.norm_out = Normalize(current_channels)
        self.conv_out = nn.Conv2d(current_channels, latent_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Forward pass for the Encoder.

        Args:
            x (torch.Tensor):
                Input tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor:
                The latent representation of shape:
                (B, z_channels, H_out, W_out) or (B, 2*z_channels, H_out, W_out),
                where H_out and W_out are spatially downsampled versions of H, W.
        """
        # Initial projection
        h = self.initial_conv(x)

        # Downsampling through each resolution
        for down in self.down_blocks:
            # Apply ResNet blocks and attention
            for res_block, attn_block in zip(down.block, down.attn):
                h = res_block(h)
                h = attn_block(h)

            # Downsample if defined
            if hasattr(down, "downsample"):
                h = down.downsample(h)

        # Middle block
        h = self.middle.block_1(h)
        h = self.middle.attn_1(h)
        h = self.middle.block_2(h)

        # Final projection
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)

        return h


class Upsample2x(nn.Module):
    """
    A simple 2x upsampling module using nearest-neighbor interpolation,
    followed by a 3x3 convolution to refine features.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_upsampled = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x_upsampled)


class Decoder(nn.Module):
    """
    Decoder that reconstructs images from latent codes via a series of
    ResNet blocks, and 2x upsampling steps.
    inspired by https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py#L368

    Args:
        base_channels (int):
            Number of base channels in the network. All channel multipliers
            scale relative to this base value.
        channel_multipliers (Tuple[int]):
            Sequence of multipliers that define how many channels we have
            at each resolution stage. E.g. (1, 2, 4, 8).
        num_resnet_blocks (int):
            Number of ResNet blocks at each resolution. In the decoder,
            we actually do (num_resnet_blocks + 1) blocks per stage.
        dropout_rate (float):
            Dropout probability. If <= 1e-6, effectively no dropout is applied.
        in_channels (int):
            Number of output channels, e.g. 3 for RGB reconstruction.
        latent_channels (int):
            Number of channels in the input latent representation.
        use_self_attn (bool):
            Whether to apply self-attention at the bottom (lowest resolution).
        use_mid_self_attn (bool):
            Whether to apply self-attention in the middle block.

    Shape:
        - Input: (B, latent_channels, H_latent, W_latent)
        - Output: (B, in_channels, H_out, W_out)

      Typically, H_out = H_latent * 2^(num_channel_multipliers),
      W_out = W_latent * 2^(num_channel_multipliers).

    Example:
        >>> decoder = Decoder(
        ...     base_channels=128,
        ...     channel_multipliers=(1, 2, 4, 8),
        ...     num_resnet_blocks=2,
        ...     dropout_rate=0.0,
        ...     in_channels=3,
        ...     latent_channels=256,
        ...     use_self_attn=True,
        ...     use_mid_self_attn=True,
        ... )
        >>> latents = torch.randn(4, 256, 8, 8)  # e.g. batch of 4
        >>> recon = decoder(latents)
        >>> print(recon.shape)
        torch.Size([4, 3, 64, 64])
    """

    def __init__(
        self,
        *,
        base_channels=128,  # Base number of feature maps
        channel_multipliers=(1, 2, 4, 8),  # How channels scale at each resolution
        num_resnet_blocks=2,  # Number of ResNet blocks per resolution
        dropout_rate=0.0,
        in_channels=3,  # Number of channels in output images (e.g. 3 for RGB)
        latent_channels,  # Number of channels in the latent representation
    ):
        super().__init__()

        # Store config
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_resnet_blocks = num_resnet_blocks
        self.dropout_rate = dropout_rate
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        # Number of resolution levels
        self.num_resolution_levels = len(channel_multipliers)

        # Starting channel count at the lowest (bottleneck) resolution
        current_in_channels = base_channels * channel_multipliers[-1]

        # 1) Latent -> First convolution to get to 'current_in_channels'
        self.latent_to_features = nn.Conv2d(latent_channels, current_in_channels, kernel_size=3, stride=1, padding=1)

        # 2) Middle block
        self.middle_block = nn.Module()
        self.middle_block.resnet_1 = ResnetBlock(
            in_channels=current_in_channels,
            out_channels=current_in_channels,
            dropout_rate=dropout_rate,
        )
        self.middle_block.attn_1 = SelfAttention(current_in_channels)
        self.middle_block.resnet_2 = ResnetBlock(
            in_channels=current_in_channels,
            out_channels=current_in_channels,
            dropout_rate=dropout_rate,
        )

        # 3) Upsampling stages: we go from bottom (lowest resolution) to top (highest)
        self.upsample_stages = nn.ModuleList()

        for level_idx in reversed(range(self.num_resolution_levels)):
            stage = nn.Module()

            stage.resnet_blocks = nn.ModuleList()
            stage.attn_blocks = nn.ModuleList()

            out_channels_at_level = base_channels * channel_multipliers[level_idx]

            # We do (num_resnet_blocks + 1) blocks at each resolution
            for block_idx in range(self.num_resnet_blocks + 1):
                resnet_block = ResnetBlock(
                    in_channels=current_in_channels,
                    out_channels=out_channels_at_level,
                    dropout_rate=dropout_rate,
                )
                stage.resnet_blocks.append(resnet_block)

                # Update current_in_channels after each block
                current_in_channels = out_channels_at_level

                # If we're at the bottom level (lowest resolution),
                # add an attention block for each ResNet block.
                if level_idx == self.num_resolution_levels - 1:
                    stage.attn_blocks.append(SelfAttention(current_in_channels))
                else:
                    stage.attn_blocks.append(nn.Identity())

            # Upsample after each level except the last (level_idx=0 is the top)
            if level_idx != 0:
                stage.upsample = Upsample2x(current_in_channels)
            else:
                stage.upsample = nn.Identity()

            self.upsample_stages.insert(0, stage)  # prepend to keep stage order in [low->high]

        # 4) Final normalization + output conv
        self.final_norm = Normalize(current_in_channels)
        self.final_conv = nn.Conv2d(current_in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, latent_codes):
        """
        Forward pass of the Decoder.

        Args:
            latent_codes: A latent tensor of shape (B, latent_channels, H_latent, W_latent).

        Returns:
            Reconstructed images of shape (B, in_channels, H_out, W_out),
            where typically H_out and W_out are H_latent and W_latent multiplied
            by 2^(num_resolution_levels).
        """

        # 1) Map latents to feature space
        hidden_states = self.latent_to_features(latent_codes)

        # 2) Middle block
        hidden_states = self.middle_block.resnet_1(hidden_states)
        hidden_states = self.middle_block.attn_1(hidden_states)
        hidden_states = self.middle_block.resnet_2(hidden_states)

        # 3) Upsampling through each resolution stage
        #    (in reverse order compared to the encoder's downsampling)
        for stage_idx in reversed(range(self.num_resolution_levels)):
            stage = self.upsample_stages[stage_idx]
            for block_idx in range(self.num_resnet_blocks + 1):
                hidden_states = stage.resnet_blocks[block_idx](hidden_states)
                hidden_states = stage.attn_blocks[block_idx](hidden_states)
            hidden_states = stage.upsample(hidden_states)

        # 4) Final output projection
        hidden_states = self.final_conv(F.silu(self.final_norm(hidden_states), inplace=True))
        return hidden_states


class TestModels(unittest.TestCase):

    def test_normalize(self):
        batch_size, channels, H, W = 2, 4, 8, 8
        x = torch.randn(batch_size, channels, H, W)
        norm_layer = Normalize(in_channels=channels)
        out = norm_layer(x)
        self.assertEqual(x.shape, out.shape)

    def test_resnetblock_identity(self):
        # Test ResnetBlock where in_channels == out_channels
        batch_size, in_channels, H, W = 2, 3, 16, 16
        x = torch.randn(batch_size, in_channels, H, W)
        block = ResnetBlock(in_channels=in_channels, out_channels=in_channels, dropout_rate=0.0)
        out = block(x)
        self.assertEqual(x.shape, out.shape)

    def test_resnetblock_projection(self):
        # Test ResnetBlock where in_channels != out_channels
        batch_size, in_channels, out_channels, H, W = 2, 3, 6, 16, 16
        x = torch.randn(batch_size, in_channels, H, W)
        block = ResnetBlock(in_channels=in_channels, out_channels=out_channels, dropout_rate=0.0)
        out = block(x)
        self.assertEqual(out.shape, (batch_size, out_channels, H, W))

    def test_self_attention(self):
        batch_size, channels, H, W = 2, 8, 16, 16
        x = torch.randn(batch_size, channels, H, W)
        attn = SelfAttention(embed_dim=channels)
        out = attn(x)
        self.assertEqual(out.shape, x.shape)

    def test_make_attention_block(self):
        block_true = make_attention_block(8, use_self_attn=True)
        self.assertIsInstance(block_true, SelfAttention)
        block_false = make_attention_block(8, use_self_attn=False)
        # nn.Identity does not have parameters so checking type is sufficient.
        self.assertTrue(isinstance(block_false, nn.Identity))

    def test_downsample2x(self):
        batch_size, channels, H, W = 2, 3, 32, 32
        x = torch.randn(batch_size, channels, H, W)
        downsample = Downsample2x(in_channels=channels)
        out = downsample(x)
        # After padding and a stride-2 convolution, 32 -> 16
        self.assertEqual(out.shape, (batch_size, channels, 16, 16))

    def test_encoder(self):
        batch_size, in_channels, H, W = 4, 3, 64, 64
        latent_channels = 256
        encoder = Encoder(
            base_channels=128,
            channel_multipliers=(1, 2, 4, 8),
            num_resnet_blocks=2,
            dropout_rate=0.0,
            in_channels=in_channels,
            latent_channels=latent_channels,
        )
        x = torch.randn(batch_size, in_channels, H, W)
        out = encoder(x)
        # With 4 resolution levels, downsampling is applied 3 times (64 -> 32 -> 16 -> 8)
        self.assertEqual(out.shape, (batch_size, latent_channels, 8, 8))

    def test_upsample2x(self):
        batch_size, channels, H, W = 2, 64, 8, 8
        x = torch.randn(batch_size, channels, H, W)
        upsample = Upsample2x(in_channels=channels)
        out = upsample(x)
        self.assertEqual(out.shape, (batch_size, channels, H * 2, W * 2))

    def test_decoder(self):
        batch_size, latent_channels, H, W = 4, 256, 8, 8
        decoder = Decoder(
            base_channels=128,
            channel_multipliers=(1, 2, 4, 8),
            num_resnet_blocks=2,
            dropout_rate=0.0,
            in_channels=3,
            latent_channels=latent_channels,
        )
        latents = torch.randn(batch_size, latent_channels, H, W)
        recon = decoder(latents)
        # With 4 resolution levels, upsampling is applied 3 times (8 -> 16 -> 32 -> 64)
        self.assertEqual(recon.shape, (batch_size, 3, 64, 64))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 64
    num_epochs = 5
    learning_rate = 1e-3

    transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )

    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    encoder = Encoder(
        base_channels=32,
        channel_multipliers=(1, 2, 4),
        num_resnet_blocks=2,
        dropout_rate=0.0,
        in_channels=3,
        latent_channels=16,
    ).to(device)

    decoder = Decoder(
        base_channels=32,
        channel_multipliers=(1, 2, 4),
        num_resnet_blocks=2,
        dropout_rate=0.0,
        in_channels=3,
        latent_channels=16,
    ).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        running_loss = 0.0

        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()
            latents = encoder(images)
            outputs = decoder(latents)

            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}")

        encoder.eval()
        decoder.eval()
        test_loss = 0.0

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                latents = encoder(images)
                outputs = decoder(latents)
                loss = criterion(outputs, images)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] Test Loss: {avg_test_loss:.4f}")

    print("Training complete.")

    # Plot training and test loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss", marker="o")
    plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import matplotlib.pyplot as plt

    # unittest.main()

    main()
