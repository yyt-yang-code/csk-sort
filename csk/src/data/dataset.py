"""
EcoSort Dataset Module
Implementation of a classification dataset supporting dynamic classes 
and pre-partitioned directory structures.
"""

import os
from typing import Callable, Dict, List, Optional, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class TrashDataset(Dataset):
    """Waste Classification Dataset

    Supports two directory structures:
    1) Unstructured: root/class_name/*.jpg (Randomly split into train/val via val_split)
    2) Structured: root/train|val|test/class_name/*.jpg (Loaded directly based on split)
    """

    # Default classes for backward compatibility
    DEFAULT_CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    CLASS_NAMES = DEFAULT_CLASS_NAMES

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        split: str = 'train',
        val_split: float = 0.2,
        seed: int = 42,
        class_names: Optional[List[str]] = None
    ):
        """
        Args:
            root_dir: Dataset root directory containing class_name/xxx.jpg structure
            transform: Image transformation pipeline
            split: Target subset - 'train', 'val', or 'test'
            val_split: Ratio of data to use for validation
            seed: Random seed for reproducibility
            class_names: Explicit list of class names
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.class_names = class_names
        self.class_to_idx = {}
        self.samples = []
        self.targets = []

        self._split_root = self.root_dir / split
        self.using_pre_split = self._split_root.exists() and self._split_root.is_dir()

        self._initialize_classes()

        # Load all dataset samples
        self._load_samples()

        # Handle data partitioning
        if not self.using_pre_split:
            self._split_data(val_split, seed)

        print(f"[{split.upper()}] Loaded {len(self.samples)} samples "
              f"across {len(self.class_names)} classes")

    def _initialize_classes(self):
        """Initialize the class list and mapping"""
        if self.class_names is not None:
            self.class_names = list(self.class_names)
        else:
            search_root = self._split_root if self.using_pre_split else self.root_dir
            discovered = sorted([
                p.name for p in search_root.iterdir()
                if p.is_dir() and not p.name.startswith('.')
            ]) if search_root.exists() else []

            if discovered:
                self.class_names = discovered
            else:
                self.class_names = list(self.DEFAULT_CLASS_NAMES)

        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def _load_samples(self):
        """Map all image file paths to their respective labels"""
        base_dir = self._split_root if self.using_pre_split else self.root_dir

        for class_name in self.class_names:
            class_dir = base_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Directory {class_dir} not found. Skipping...")
                continue

            # Support multiple image extensions
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                for img_path in class_dir.glob(ext):
                    self.samples.append(str(img_path))
                    self.targets.append(self.class_to_idx[class_name])

        if len(self.samples) == 0:
            raise ValueError(f"No valid image files found in {self.root_dir}")

    def _split_data(self, val_split: float, seed: int):
        """Partition data into training, validation, or test sets"""
        np.random.seed(seed)
        indices = np.arange(len(self.samples))
        np.random.shuffle(indices)

        # Calculate split sizes
        val_size = int(len(indices) * val_split)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        if self.split == 'train':
            selected_indices = train_indices
        elif self.split == 'val':
            selected_indices = val_indices
        else:  # 'test' mode uses the full dataset indices
            selected_indices = indices

        # Update sample and target lists
        self.samples = [self.samples[i] for i in selected_indices]
        self.targets = [self.targets[i] for i in selected_indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: (C, H, W) tensor
            label: Integer label index
        """
        img_path = self.samples[idx]
        label = self.targets[idx]

        # Attempt to load the image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
            # Fallback to a blank white image to avoid crashing the training loop
            image = Image.new('RGB', (256, 256), color='white')

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """Return the count of samples per class"""
        distribution = {name: 0 for name in self.class_names}
        for label in self.targets:
            class_name = self.class_names[label]
            distribution[class_name] += 1
        return distribution


def get_data_transforms(
    mode: str = 'train',
    img_size: int = 256,
    strong_aug: bool = False
) -> transforms.Compose:
    """Utility for building image transformation pipelines

    Args:
        mode: 'train', 'val', or 'test'
        img_size: Target resolution for resizing
        strong_aug: Enable advanced augmentation (recommended for 42-class fine-tuning)

    Returns:
        transforms.Compose pipeline
    """
    # Standard ImageNet normalization parameters
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if mode == 'train':
        if strong_aug:
            # Advanced augmentation pipeline for fine-grained classification
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
            ])
        else:
            # Standard training augmentation
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                normalize,
            ])
    else:
        # Static pipeline for validation and testing
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])


def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 256,
    val_split: float = 0.2,
    class_names: Optional[List[str]] = None,
    strong_aug: bool = False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Factory function for creating Train and Validation DataLoaders

    Args:
        data_root: Path to dataset
        batch_size: Number of samples per batch
        num_workers: Data loading parallel processes
        img_size: Input image resolution
        val_split: Validation partition ratio
        class_names: Explicit labels
        strong_aug: Toggle for advanced augmentation

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = TrashDataset(
        root_dir=data_root,
        transform=get_data_transforms('train', img_size, strong_aug=strong_aug),
        split='train',
        val_split=val_split,
        class_names=class_names
    )

    val_dataset = TrashDataset(
        root_dir=data_root,
        transform=get_data_transforms('val', img_size),
        split='val',
        val_split=val_split,
        class_names=train_dataset.class_names
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == '__main__':
    # Unit Testing
    dataset = TrashDataset(
        root_dir='data/raw',
        transform=get_data_transforms('train'),
        split='train'
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")

    # Sample Verification
    img, label = dataset[0]
    print(f"Image tensor shape: {img.shape}, Label index: {label}, "
          f"Category: {dataset.class_names[label]}")
