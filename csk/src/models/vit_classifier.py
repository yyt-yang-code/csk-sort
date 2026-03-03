"""Vision Transformer classifier for EcoSort.

This module provides a drop-in ViT implementation compatible with the
existing training entrypoint interface used across the project.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torchvision.models as models


class ViTClassifier(nn.Module):
    """Vision Transformer classifier wrapper.

    Supported backbones (subject to torchvision version support):
    - vit_b_16
    - vit_b_32
    - vit_l_16
    - vit_l_32
    - vit_h_14
    """

    _MODEL_BUILDERS: Dict[str, str] = {
        'vit_b_16': 'vit_b_16',
        'vit_b_32': 'vit_b_32',
        'vit_l_16': 'vit_l_16',
        'vit_l_32': 'vit_l_32',
        'vit_h_14': 'vit_h_14',
    }

    _WEIGHTS_ENUMS: Dict[str, str] = {
        'vit_b_16': 'ViT_B_16_Weights',
        'vit_b_32': 'ViT_B_32_Weights',
        'vit_l_16': 'ViT_L_16_Weights',
        'vit_l_32': 'ViT_L_32_Weights',
        'vit_h_14': 'ViT_H_14_Weights',
    }

    def __init__(
        self,
        num_classes: int = 4,
        backbone: str = 'vit_b_16',
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        """Initialize a ViT classifier.

        Args:
            num_classes: Number of target classes.
            backbone: ViT backbone name.
            pretrained: Whether to load ImageNet pretrained weights.
            dropout: Dropout probability applied to the classification head.
        """
        super().__init__()

        if backbone not in self._MODEL_BUILDERS:
            supported = ', '.join(sorted(self._MODEL_BUILDERS.keys()))
            raise ValueError(f"Unsupported ViT backbone: {backbone}. Supported: {supported}")

        self.num_classes = num_classes
        self.backbone_name = backbone

        self.backbone = self._build_backbone(
            backbone=backbone,
            pretrained=pretrained,
        )

        # Replace classifier head to fit project classes and configurable dropout.
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        self._init_classifier_weights()

    def _build_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
        """Build torchvision ViT backbone with version-safe weight loading."""
        builder_name = self._MODEL_BUILDERS[backbone]
        builder = getattr(models, builder_name, None)
        if builder is None:
            raise RuntimeError(
                f"Current torchvision does not provide '{builder_name}'. "
                "Please upgrade torchvision or switch to another backbone."
            )

        if not pretrained:
            return builder(weights=None)

        # New torchvision API (weights enum)
        weights_enum_name = self._WEIGHTS_ENUMS.get(backbone)
        if weights_enum_name is not None and hasattr(models, weights_enum_name):
            weights_enum = getattr(models, weights_enum_name)
            return builder(weights=weights_enum.DEFAULT)

        # Fallback for older torchvision versions.
        try:
            return builder(pretrained=True)
        except TypeError:
            return builder(weights=None)

    def _init_classifier_weights(self):
        """Initialize the custom classification head weights."""
        for module in self.backbone.heads.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return class logits."""
        return self.backbone(x)

    @torch.no_grad()
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled penultimate features for analysis tasks."""
        encoded = self.backbone._process_input(x)
        batch_size = encoded.shape[0]

        class_token = self.backbone.class_token.expand(batch_size, -1, -1)
        encoded = torch.cat([class_token, encoded], dim=1)
        encoded = self.backbone.encoder(encoded)

        # CLS token representation
        return encoded[:, 0]


def create_vit_model(
    backbone: str = 'vit_b_16',
    num_classes: int = 4,
    pretrained: bool = True,
    **kwargs,
) -> ViTClassifier:
    """Factory function to create a ViT classifier.

    Args:
        backbone: ViT backbone name.
        num_classes: Number of target classes.
        pretrained: Whether to use pretrained weights.
        **kwargs: Additional keyword arguments such as dropout.

    Returns:
        Configured ViTClassifier instance.
    """
    return ViTClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        **kwargs,
    )


if __name__ == '__main__':
    model = create_vit_model(
        backbone='vit_b_16',
        num_classes=14,
        pretrained=False,
        dropout=0.1,
    )

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    sample = torch.randn(2, 3, 224, 224)
    logits = model(sample)
    features = model.get_features(sample)

    print(f"Input shape: {sample.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
