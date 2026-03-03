"""
Letterbox Transformation
Aspect ratio preserving resize with padding
Maintains original aspect ratio by resizing the longest side to target size 
and padding the remaining area with constant color
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np


class ResizeLongestSideWithPadding:
    """Resize longest side to target size while maintaining aspect ratio, 
    pad remaining area to create square image
    
    This transformation preserves the original aspect ratio by:
    1. Scaling the image so the longest side matches target size
    2. Padding the shorter sides with constant color to create square output
    """

    def __init__(self, target_size: int = 224, fill_color: tuple = (114, 114, 114)):
        """
        Args:
            target_size: Desired output size (square dimension)
            fill_color: RGB color tuple for padding (default: (114, 114, 114))
        """
        self.target_size = target_size
        self.fill_color = fill_color

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply letterbox transformation to image
        
        Args:
            img: Input PIL Image
            
        Returns:
            Transformed PIL Image with target size (square)
        """
        # Get original dimensions
        w, h = img.size

        # Calculate scaling factor to fit longest side to target size
        scale = self.target_size / max(w, h)

        # Compute new dimensions with preserved aspect ratio
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize image with bilinear interpolation
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        # Create blank canvas with target size and fill color
        img_padded = Image.new('RGB', (self.target_size, self.target_size), self.fill_color)

        # Center the resized image on the canvas
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        img_padded.paste(img_resized, (paste_x, paste_y))

        return img_padded


def test_letterbox():
    """Test letterbox transformation functionality"""
    # Create test image (640x480 red image)
    img = Image.new('RGB', (640, 480), color='red')

    # Apply letterbox transformation
    transform = ResizeLongestSideWithPadding(target_size=224)
    img_transformed = transform(img)

    print(f"Original size: {img.size}")
    print(f"Transformed size: {img_transformed.size}")

    assert img_transformed.size == (224, 224), "Incorrect output dimensions"
    print("✅ Letterbox transformation test passed")


if __name__ == '__main__':
    test_letterbox()