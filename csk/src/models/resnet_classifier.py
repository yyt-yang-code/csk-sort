"""
EcoSort ResNet Classifier
Trash classification model based on ResNet architecture with attention mechanism
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ResNetClassifier(nn.Module):
    """ResNet-based Trash Classification Model

    Supported backbones:
    - resnet50
    - resnet101
    """

    def __init__(
        self,
        num_classes: int = 4,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.3,
        use_attention: bool = False
    ):
        """
        Args:
            num_classes: Number of classification categories (4 trash types)
            backbone: ResNet variant to use
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout probability for regularization
            use_attention: Whether to incorporate CBAM attention mechanism
        """
        super(ResNetClassifier, self).__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone
        self.use_attention = use_attention

        # Load pretrained ResNet backbone
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Extract feature extractor (remove final fully connected layer)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # Optional attention mechanism
        if use_attention:
            self.attention = CBAM(self.feature_dim)
        else:
            self.attention = None

        # Global average pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )

        # Initialize classifier weights
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """Initialize classification head weights using He initialization"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the network

        Args:
            x: Input tensor with shape (B, 3, H, W)

        Returns:
            logits: Classification logits with shape (B, num_classes)
        """
        # Feature extraction
        features = self.features(x)  # (B, 2048, H/32, W/32)

        # Apply attention mechanism if enabled
        if self.attention is not None:
            features = self.attention(features)

        # Global average pooling
        pooled = self.avgpool(features)  # (B, 2048, 1, 1)

        # Classification
        logits = self.classifier(pooled)

        return logits

    def get_features(self, x):
        """Extract intermediate features (for visualization or t-SNE analysis)"""
        features = self.features(x)
        if self.attention is not None:
            features = self.attention(features)
        pooled = self.avgpool(features)
        return pooled.view(pooled.size(0), -1)


class CBAM(nn.Module):
    """Convolutional Block Attention Module
    Reference: CBAM: Convolutional Block Attention Module (ECCV 2018)
    """

    def __init__(self, channels: int, reduction_ratio: int = 16):
        super(CBAM, self).__init__()

        # Channel attention mechanism
        self.channel_attention = SequentialPolarized(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=True)
        )

        # Spatial attention mechanism
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor with shape (B, C, H, W)

        Returns:
            out: Feature tensor with attention weighting (B, C, H, W)
        """
        # Channel attention
        ca = self.channel_attention(x)
        ca = torch.sigmoid(ca)
        x = x * ca

        # Spatial attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_input = torch.cat([max_pool, avg_pool], dim=1)
        sa = self.spatial_attention(spatial_input)
        x = x * sa

        return x


class SequentialPolarized(nn.Module):
    """Simplified sequential module for channel attention implementation"""
    def __init__(self, *args):
        super(SequentialPolarized, self).__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x


def create_resnet_model(
    backbone: str = 'resnet50',
    num_classes: int = 4,
    pretrained: bool = True,
    **kwargs
) -> ResNetClassifier:
    """Factory function to create ResNet classifier instances

    Args:
        backbone: ResNet architecture type
        num_classes: Number of classification categories
        pretrained: Whether to use ImageNet pretrained weights
        **kwargs: Additional model configuration parameters

    Returns:
        model: Initialized ResNetClassifier instance
    """
    model = ResNetClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,** kwargs
    )

    return model


if __name__ == '__main__':
    # Model testing
    model = create_resnet_model(
        backbone='resnet50',
        num_classes=4,
        pretrained=False,
        use_attention=True
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
