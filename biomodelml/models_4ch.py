"""
4-Channel Siamese Regressor for Evolutionary Distance Prediction

This module implements a Siamese neural network that accepts 4-channel input:
- Channels 0-2: RGB (normalized to [-1, 1])
- Channel 3: Binary mask (1.0 = valid sequence data, 0.0 = padding)

The mask channel allows the model to explicitly learn to ignore padded regions,
maintaining the "1 pixel = 1 residue interaction" biochemical scale without
confusion from zero-padding artifacts.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SiameseRegressor4Channel(nn.Module):
    """
    A Siamese Neural Network for evolutionary distance regression with 4-channel input.

    This model extends the standard 3-channel Siamese architecture by adding a binary
    mask channel that explicitly marks valid sequence data vs padding. The first
    convolutional layer is modified to accept 4 channels while preserving pretrained
    ImageNet weights for the RGB channels.

    Architecture:
        1. Shared backbone (ResNet-50 or EfficientNet-B0) with 4-channel input
        2. Global average pooling
        3. Feature fusion: |v1 - v2| (absolute difference)
        4. Regression head: 3 Dense layers (512 → 128 → 1)
    """

    def __init__(self, backbone='resnet50', pretrained=True, freeze_backbone=True,
                 mask_init_mode='zero'):
        """
        Initializes the 4-Channel Siamese Regressor.

        Args:
            backbone (str): The name of the backbone model ('resnet50' or 'efficientnet_b0').
            pretrained (bool): Whether to use pre-trained ImageNet weights for RGB channels.
            freeze_backbone (bool): If True, freeze the weights of the backbone.
            mask_init_mode (str): How to initialize the 4th channel weights:
                - 'zero': Initialize to 0 (preserves pretrained features, learn gradually)
                - 'mean': Initialize to mean of RGB channels (faster convergence)
                - 'random': Random initialization with Kaiming normal
        """
        super(SiameseRegressor4Channel, self).__init__()

        self.backbone_name = backbone
        self.mask_init_mode = mask_init_mode

        # Load and modify the backbone for 4-channel input
        if backbone == 'resnet50':
            self.backbone, num_features = self._create_resnet50_4ch(pretrained, mask_init_mode)
        elif backbone == 'efficientnet_b0':
            self.backbone, num_features = self._create_efficientnet_4ch(pretrained, mask_init_mode)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'resnet50' or 'efficientnet_b0'")

        # Freeze backbone layers if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"Backbone ({backbone}) frozen. Only the regression head will be trained.")

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Regression head (same as 3-channel version)
        self.regression_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU()  # Ensure the output is non-negative
        )

    def _create_resnet50_4ch(self, pretrained, mask_init_mode):
        """
        Create ResNet-50 backbone with 4-channel input.

        Modifies the first convolutional layer from Conv2d(3, 64, ...) to Conv2d(4, 64, ...)
        while preserving pretrained weights for RGB channels.

        Args:
            pretrained (bool): Whether to use ImageNet weights
            mask_init_mode (str): How to initialize the 4th channel

        Returns:
            tuple: (modified_backbone, num_features)
        """
        # Load pretrained ResNet-50
        base_model = models.resnet50(pretrained=pretrained)

        # Get the original first conv layer (3 input channels)
        original_conv1 = base_model.conv1  # Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Create new conv layer with 4 input channels
        new_conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Copy pretrained weights for RGB channels (channels 0-2)
        with torch.no_grad():
            new_conv1.weight[:, :3, :, :] = original_conv1.weight.clone()

            # Initialize the 4th channel (mask) based on mask_init_mode
            if mask_init_mode == 'zero':
                # Zero initialization: model learns mask importance from scratch
                new_conv1.weight[:, 3:4, :, :] = 0.0
            elif mask_init_mode == 'mean':
                # Mean of RGB: assumes mask has similar importance to RGB
                new_conv1.weight[:, 3:4, :, :] = original_conv1.weight.mean(dim=1, keepdim=True)
            elif mask_init_mode == 'random':
                # Random initialization: standard Kaiming normal
                nn.init.kaiming_normal_(new_conv1.weight[:, 3:4, :, :], mode='fan_out', nonlinearity='relu')
            else:
                raise ValueError(f"Invalid mask_init_mode: {mask_init_mode}")

        # Replace the first conv layer in the base model
        base_model.conv1 = new_conv1

        # Get number of features before final FC layer
        num_features = base_model.fc.in_features

        # Remove the final fully connected layer (classifier)
        backbone = nn.Sequential(*list(base_model.children())[:-1])

        return backbone, num_features

    def _create_efficientnet_4ch(self, pretrained, mask_init_mode):
        """
        Create EfficientNet-B0 backbone with 4-channel input.

        Modifies the first convolutional layer to accept 4 channels while preserving
        pretrained weights for RGB channels.

        Args:
            pretrained (bool): Whether to use ImageNet weights
            mask_init_mode (str): How to initialize the 4th channel

        Returns:
            tuple: (modified_backbone, num_features)
        """
        # Load pretrained EfficientNet-B0
        base_model = models.efficientnet_b0(pretrained=pretrained)

        # The first conv layer is in features[0][0]
        original_conv = base_model.features[0][0]  # Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Create new conv layer with 4 input channels
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        # Copy pretrained weights for RGB channels
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = original_conv.weight.clone()

            # Initialize the 4th channel based on mask_init_mode
            if mask_init_mode == 'zero':
                new_conv.weight[:, 3:4, :, :] = 0.0
            elif mask_init_mode == 'mean':
                new_conv.weight[:, 3:4, :, :] = original_conv.weight.mean(dim=1, keepdim=True)
            elif mask_init_mode == 'random':
                nn.init.kaiming_normal_(new_conv.weight[:, 3:4, :, :], mode='fan_out', nonlinearity='relu')
            else:
                raise ValueError(f"Invalid mask_init_mode: {mask_init_mode}")

        # Replace the first conv layer
        base_model.features[0][0] = new_conv

        # Get number of features
        num_features = base_model.classifier[1].in_features

        # Remove classifier
        base_model.classifier = nn.Identity()

        return base_model, num_features

    def _forward_one(self, x):
        """
        Passes one input through the backbone and pooling layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, 4, H, W)

        Returns:
            torch.Tensor: Feature vector of shape (batch, num_features)
        """
        x = self.backbone(x)

        # EfficientNet already includes global pooling and flattening
        if self.backbone_name == 'efficientnet_b0':
            return x

        # ResNet needs explicit pooling and flattening
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x1, x2):
        """
        The forward pass of the Siamese network.

        Args:
            x1 (torch.Tensor): First input tensor of shape (batch, 4, H, W)
            x2 (torch.Tensor): Second input tensor of shape (batch, 4, H, W)

        Returns:
            torch.Tensor: Predicted evolutionary distance of shape (batch, 1)
        """
        # Validate input shape
        if x1.shape[1] != 4 or x2.shape[1] != 4:
            raise ValueError(f"Expected 4 channels, got x1: {x1.shape[1]}, x2: {x2.shape[1]}")

        # Extract feature vectors for both inputs
        v1 = self._forward_one(x1)
        v2 = self._forward_one(x2)

        # Calculate absolute difference (feature fusion)
        v_diff = torch.abs(v1 - v2)

        # Pass through regression head
        output = self.regression_head(v_diff)

        return output

    def get_mask_channel_stats(self):
        """
        Get statistics about the mask channel weights to verify it's being used.

        Returns:
            dict: Statistics including min, max, mean, std of mask channel weights
        """
        if self.backbone_name == 'resnet50':
            mask_weights = self.backbone[0].weight[:, 3:4, :, :].detach().cpu()
        elif self.backbone_name == 'efficientnet_b0':
            mask_weights = self.backbone.features[0][0].weight[:, 3:4, :, :].detach().cpu()
        else:
            return {}

        return {
            'min': mask_weights.min().item(),
            'max': mask_weights.max().item(),
            'mean': mask_weights.mean().item(),
            'std': mask_weights.std().item(),
            'abs_mean': mask_weights.abs().mean().item(),
        }


if __name__ == '__main__':
    print("=" * 80)
    print("Testing SiameseRegressor4Channel with ResNet-50 backbone")
    print("=" * 80)

    # Test ResNet-50 with 4 channels
    print("\n1. Initializing model with mask_init_mode='zero'...")
    model_zero = SiameseRegressor4Channel(
        backbone='resnet50',
        pretrained=True,
        freeze_backbone=False,  # Allow gradients for testing
        mask_init_mode='zero'
    )
    print(f"Model created successfully!")

    # Check mask channel initialization
    print("\n2. Checking mask channel initialization...")
    stats = model_zero.get_mask_channel_stats()
    print(f"Mask channel stats (zero init): {stats}")
    assert abs(stats['abs_mean']) < 1e-6, "Zero init should have weights near 0"
    print("✓ Zero initialization verified")

    # Create dummy 4-channel input
    print("\n3. Testing forward pass with dummy input...")
    batch_size = 4
    channels = 4
    height = 512
    width = 512

    # Create dummy tensors (RGB normalized to [-1,1], mask in [0,1])
    dummy_rgb = torch.randn(batch_size, 3, height, width)  # RGB channels
    dummy_mask = torch.ones(batch_size, 1, height, width) * 0.8  # Mask channel
    dummy_input1 = torch.cat([dummy_rgb, dummy_mask], dim=1)

    dummy_rgb2 = torch.randn(batch_size, 3, height, width)
    dummy_mask2 = torch.ones(batch_size, 1, height, width) * 0.6
    dummy_input2 = torch.cat([dummy_rgb2, dummy_mask2], dim=1)

    print(f"Input tensor shape: {dummy_input1.shape}")
    print(f"RGB range: [{dummy_input1[:, :3].min():.2f}, {dummy_input1[:, :3].max():.2f}]")
    print(f"Mask range: [{dummy_input1[:, 3].min():.2f}, {dummy_input1[:, 3].max():.2f}]")

    # Forward pass
    with torch.no_grad():
        output = model_zero(dummy_input1, dummy_input2)

    print(f"\n4. Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, 1)")
    assert output.shape == (batch_size, 1), f"Shape mismatch: {output.shape}"
    print("✓ Output shape correct")

    print(f"\n5. Example predictions: {output.squeeze().tolist()}")
    assert not torch.isnan(output).any(), "Output contains NaN!"
    assert not torch.isinf(output).any(), "Output contains Inf!"
    print("✓ No NaN or Inf in output")

    # Test gradient flow
    print("\n6. Testing gradient flow...")
    model_zero.train()
    criterion = nn.MSELoss()
    target = torch.rand(batch_size, 1)

    output = model_zero(dummy_input1, dummy_input2)
    loss = criterion(output, target)
    loss.backward()

    # Check if gradients exist for regression head
    head_params_with_grad = sum(p.grad is not None for p in model_zero.regression_head.parameters())
    print(f"Regression head parameters with gradients: {head_params_with_grad}")
    assert head_params_with_grad > 0, "No gradients in regression head!"
    print("✓ Gradients flowing correctly")

    # Test with different mask_init_modes
    print("\n" + "=" * 80)
    print("Testing different mask initialization modes")
    print("=" * 80)

    for init_mode in ['zero', 'mean', 'random']:
        print(f"\nTesting mask_init_mode='{init_mode}'...")
        model_test = SiameseRegressor4Channel(
            backbone='resnet50',
            pretrained=True,
            freeze_backbone=True,
            mask_init_mode=init_mode
        )
        stats = model_test.get_mask_channel_stats()
        print(f"  Stats: mean={stats['mean']:.6f}, std={stats['std']:.6f}, abs_mean={stats['abs_mean']:.6f}")

        with torch.no_grad():
            out = model_test(dummy_input1, dummy_input2)
        print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
        print(f"✓ {init_mode} mode works!")

    # Test EfficientNet-B0
    print("\n" + "=" * 80)
    print("Testing SiameseRegressor4Channel with EfficientNet-B0 backbone")
    print("=" * 80)

    model_effnet = SiameseRegressor4Channel(
        backbone='efficientnet_b0',
        pretrained=True,
        mask_init_mode='zero'
    )
    print("EfficientNet-B0 model created successfully!")

    with torch.no_grad():
        output_effnet = model_effnet(dummy_input1, dummy_input2)

    print(f"EfficientNet output shape: {output_effnet.shape}")
    print(f"EfficientNet predictions: {output_effnet.squeeze().tolist()}")
    assert output_effnet.shape == (batch_size, 1), "EfficientNet shape mismatch"
    print("✓ EfficientNet-B0 works!")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe 4-channel Siamese regressor is ready for training.")
    print("Recommended next steps:")
    print("  1. Create datasets_4ch.py with mask generation")
    print("  2. Test with real sequence data")
    print("  3. Train on small dataset (100 samples) to verify learning")
