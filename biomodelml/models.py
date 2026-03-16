import torch
import torch.nn as nn
import torchvision.models as models

class SiameseRegressor(nn.Module):
    """
    A Siamese Neural Network for regression.
    This model takes two images as input, processes them through identical
    backbone networks (with shared weights), and predicts the distance
    (a continuous value) between them.
    """
    def __init__(self, backbone='resnet50', pretrained=True, freeze_backbone=True):
        """
        Initializes the Siamese Regressor.
        Args:
            backbone (str): The name of the backbone model (e.g., 'resnet50', 'efficientnet_b0').
            pretrained (bool): Whether to use pre-trained weights for the backbone.
            freeze_backbone (bool): If True, freeze the weights of the backbone.
        """
        super(SiameseRegressor, self).__init__()

        # 1. Load the backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            # The output features from ResNet-50's final layer before the classifier
            num_features = self.backbone.fc.in_features
            # Remove the final fully connected layer (classifier)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze backbone layers
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen. Only the regression head will be trained.")

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 2. Define the regression head
        self.regression_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU() # Ensure the output is non-negative
        )

    def _forward_one(self, x):
        """
        Passes one input through the backbone and pooling layer.
        """
        x = self.backbone(x)
        x = self.global_avg_pool(x)
        # Flatten the tensor
        x = torch.flatten(x, 1)
        return x

    def forward(self, x1, x2):
        """
        The forward pass of the Siamese network.
        Args:
            x1 (torch.Tensor): The first input image tensor.
            x2 (torch.Tensor): The second input image tensor.
        Returns:
            torch.Tensor: The predicted distance.
        """
        # 1. Get feature vectors for both inputs
        v1 = self._forward_one(x1)
        v2 = self._forward_one(x2)

        # 2. Calculate the absolute difference between the feature vectors
        v_diff = torch.abs(v1 - v2)

        # 3. Pass the difference through the regression head
        output = self.regression_head(v_diff)
        return output

if __name__ == '__main__':
    # Example of how to use the model
    print("Initializing SiameseRegressor with ResNet-50 backbone...")
    model = SiameseRegressor(backbone='resnet50')
    print(model)

    # Create dummy input tensors
    # Batch size = 4, Channels = 3, Height = 224, Width = 224 (standard for ImageNet models)
    dummy_input1 = torch.randn(4, 3, 224, 224)
    dummy_input2 = torch.randn(4, 3, 224, 224)

    print(f"\nInput tensor shape: {dummy_input1.shape}")

    # Get the model's prediction
    prediction = model(dummy_input1, dummy_input2)
    print(f"Output prediction shape: {prediction.shape}")
    print(f"Example predictions: {prediction.squeeze().tolist()}")

    # Test with EfficientNet-B0
    print("\nInitializing SiameseRegressor with EfficientNet-B0 backbone...")
    model_effnet = SiameseRegressor(backbone='efficientnet_b0')
    prediction_effnet = model_effnet(dummy_input1, dummy_input2)
    print(f"EfficientNet output prediction shape: {prediction_effnet.shape}")
