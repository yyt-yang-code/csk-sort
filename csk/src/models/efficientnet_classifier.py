"""
EcoSort EfficientNet Classifier
Efficient Waste Classification Model based on EfficientNet
"""

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from typing import Optional


class EfficientNetClassifier(nn.Module):
    """EfficientNet Waste Classifier

    Supported backbones:
    - efficientnet-b0 to efficientnet-b7
    """

    def __init__(
        self,
        num_classes: int = 4,
        backbone: str = 'efficientnet-b3',
        pretrained: bool = True,
        dropout: float = 0.3,
        drop_connect_rate: float = 0.2
    ):
        """
        Args:
            num_classes: Number of classification categories (e.g., 4 types of waste)
            backbone: EfficientNet variant
            pretrained: Whether to use ImageNet pre-trained weights
            dropout: Dropout probability
            drop_connect_rate: DropConnect probability
        """
        super(EfficientNetClassifier, self).__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone

        # Load pre-trained EfficientNet
        self.backbone = EfficientNet.from_pretrained(
            backbone if pretrained else 'efficientnet-b0',
            num_classes=num_classes,
            dropout_rate=dropout,
            drop_connect_rate=drop_connect_rate
        )

        # Get feature dimensions
        if 'b0' in backbone:
            self.feature_dim = 1280
        elif 'b1' in backbone:
            self.feature_dim = 1280
        elif 'b2' in backbone:
            self.feature_dim = 1408
        elif 'b3' in backbone:
            self.feature_dim = 1536
        elif 'b4' in backbone:
            self.feature_dim = 1792
        else:
            self.feature_dim = 1280

        # Replace the final classification layer
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )

        # Initialize classifier layer weights
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """Initialize classification head weights"""
        for m in self.backbone._fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) Input image tensor

        Returns:
            logits: (B, num_classes) Classification logits
        """
        logits = self.backbone(x)
        return logits

    def get_features(self, x):
        """Extract features (useful for visualization or t-SNE)"""
        # Extract features by removing the final classification layer
        features = self.backbone.extract_features(x)
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return pooled.view(pooled.size(0), -1)


def create_efficientnet_model(
    backbone: str = 'efficientnet-b3',
    num_classes: int = 4,
    pretrained: bool = True,
    **kwargs
) -> EfficientNetClassifier:
    """Factory function to create an EfficientNet Classifier

    Args:
        backbone: Model type variant
        num_classes: Number of classes
        pretrained: Whether to use pre-trained weights
        **kwargs: Additional parameters

    Returns:
        model: An instance of EfficientNetClassifier
    """
    model = EfficientNetClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        **kwargs
    )

    return model


if __name__ == '__main__':
    # Model Testing
    model = create_efficientnet_model(
        backbone='efficientnet-b3',
        num_classes=4,
        pretrained=False
    )

    # Calculate parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

    # Test feature extraction
    features = model.get_features(x)
    print(f"Features shape: {features.shape}")
