"""
EcoSort Training Framework
Complete training and evaluation framework for trash classification models
"""

import os
import json
import time
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import copy

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import numpy as np

from src.data.dataset import TrashDataset, get_data_transforms
from src.models.resnet_classifier import create_resnet_model
from src.models.efficientnet_classifier import create_efficientnet_model


class Trainer:
    """EcoSort Model Trainer

    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation support
    - Early stopping
    - Learning rate scheduling
    - WandB logging integration
    - Complete state checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict,
        checkpoint_dir: str = 'checkpoints',
        experiment_name: str = 'baseline',
        use_wandb: bool = True
    ):
        """
        Args:
            model: Neural network model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
            checkpoint_dir: Directory to save checkpoints
            experiment_name: Unique name for the experiment
            use_wandb: Whether to use Weights & Biases for logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # Optimizer initialization
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Loss function
        self.criterion = self._create_criterion()

        # Mixed precision training setup
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Training state tracking
        self.current_epoch = 0
        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.target_class_names = list(dict.fromkeys(config.get('target_class_names', [])))
        self.monitor_metric = str(config.get('monitor_metric', 'val_f1'))
        self.monitor_mode = str(config.get('monitor_mode', 'max'))
        
        if self.monitor_metric not in {'val_f1', 'val_acc', 'val_loss', 'val_target_f1'}:
            raise ValueError(
                f"Unsupported monitor_metric: {self.monitor_metric}. "
                "Choose from ['val_f1', 'val_acc', 'val_loss', 'val_target_f1']"
            )
        if self.monitor_mode not in {'max', 'min'}:
            raise ValueError("monitor_mode must be 'max' or 'min'")
        if self.monitor_metric == 'val_target_f1' and not self.target_class_names:
            raise ValueError(
                "monitor_metric='val_target_f1' requires training.target_class_names configuration"
            )
        
        self.best_metric = -math.inf if self.monitor_mode == 'max' else math.inf
        self.patience_counter = 0
        self.history: List[Dict[str, float]] = []

        # WandB initialization
        if use_wandb:
            import wandb
            wandb_mode = os.getenv('WANDB_MODE', 'offline')
            wandb.init(
                project='ecosort-classification',
                name=experiment_name,
                config=config,
                mode=wandb_mode
            )
            self.wandb = wandb
            print(f"WandB mode: {wandb_mode}")
        else:
            self.wandb = None

        print(f"Trainer initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Monitoring best checkpoint by {self.monitor_metric} ({self.monitor_mode})")
        if self.target_class_names:
            print(f"Target classes: {self.target_class_names}")

    def _is_better(self, current_value: float) -> bool:
        """Check if current metric value is better than best recorded"""
        if self.monitor_mode == 'max':
            return current_value > self.best_metric
        return current_value < self.best_metric

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration"""
        opt_type = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)

        if opt_type == 'adamw':
            return AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_type == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")

    def _create_scheduler(self) -> Optional[object]:
        """Create learning rate scheduler based on configuration"""
        scheduler_type = self.config.get('scheduler', 'cosine')

        if scheduler_type == 'cosine':
            epochs = self.config.get('epochs', 50)
            return CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=1e-6
            )
        elif scheduler_type == 'step':
            step_size = self.config.get('step_size', 10)
            gamma = self.config.get('gamma', 0.1)
            return StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        else:
            return None

    @staticmethod
    def _resolve_class_index(class_key: Any, class_to_idx: Dict[str, int]) -> Optional[int]:
        """Resolve class index from various key types (int/str)"""
        if isinstance(class_key, int):
            return class_key if 0 <= class_key < len(class_to_idx) else None

        class_key_str = str(class_key)
        if class_key_str in class_to_idx:
            return class_to_idx[class_key_str]

        try:
            parsed_index = int(class_key_str)
            if 0 <= parsed_index < len(class_to_idx):
                return parsed_index
        except ValueError:
            pass

        return None

    def _build_class_weights(self) -> Optional[torch.Tensor]:
        """Build class weights for imbalanced dataset training"""
        data_config = self.config.get('data', {})
        use_class_weights = data_config.get('use_class_weights', False)
        class_weight_overrides = data_config.get('class_weight_overrides', {})
        class_weight_multipliers = data_config.get('class_weight_multipliers', {})

        has_custom_weights = bool(class_weight_overrides) or bool(class_weight_multipliers)
        if not use_class_weights and not has_custom_weights:
            return None

        class_names = list(data_config.get('class_names', []))
        class_counts = list(data_config.get('class_counts', []))

        if class_names:
            num_classes = len(class_names)
        elif class_counts:
            num_classes = len(class_counts)
            class_names = [f'class_{i}' for i in range(num_classes)]
        else:
            return None

        has_valid_counts = (
            use_class_weights
            and len(class_counts) == num_classes
            and all(float(count) > 0 for count in class_counts)
        )

        if has_valid_counts:
            total = float(sum(class_counts))
            weights = torch.tensor([
                total / (num_classes * float(count))
                for count in class_counts
            ], dtype=torch.float32)
        else:
            weights = torch.ones(num_classes, dtype=torch.float32)
            if use_class_weights:
                print("\nWarning: use_class_weights=true but incomplete class_counts provided - falling back to uniform weights.")

        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        # Apply weight overrides
        for class_key, weight_value in class_weight_overrides.items():
            idx = self._resolve_class_index(class_key, class_to_idx)
            if idx is None:
                print(f"Warning: Class {class_key} not found in class_weight_overrides - skipped")
                continue

            parsed_weight = float(weight_value)
            if parsed_weight <= 0:
                print(f"Warning: Invalid weight {weight_value} for class {class_key} in class_weight_overrides - skipped")
                continue

            weights[idx] = parsed_weight

        # Apply weight multipliers
        for class_key, multiplier in class_weight_multipliers.items():
            idx = self._resolve_class_index(class_key, class_to_idx)
            if idx is None:
                print(f"Warning: Class {class_key} not found in class_weight_multipliers - skipped")
                continue

            parsed_multiplier = float(multiplier)
            if parsed_multiplier <= 0:
                print(f"Warning: Invalid multiplier {multiplier} for class {class_key} in class_weight_multipliers - skipped")
                continue

            weights[idx] = weights[idx] * parsed_multiplier

        # Normalize weights if configured
        if data_config.get('normalize_class_weights', True):
            weights = weights / weights.mean().clamp(min=1e-12)

        print(f"\nUsing class-weighted loss:")
        for idx, name in enumerate(class_names):
            count_text = (
                str(class_counts[idx])
                if idx < len(class_counts)
                else 'n/a'
            )
            print(f"  {name:20s}: count={count_text:>4s}, weight={weights[idx].item():.3f}")

        return weights.to(self.device)

    def _create_criterion(self) -> nn.Module:
        """Create loss function with optional class weighting"""
        loss_type = self.config.get('loss', {}).get('type', 'cross_entropy')
        data_config = self.config.get('data', {})
        use_weighted_sampler = data_config.get('use_weighted_sampler', False)
        allow_loss_weights_with_sampler = data_config.get('allow_loss_weights_with_sampler', False)

        class_weights = self._build_class_weights()

        # Handle weighted sampler + class weights combination
        if use_weighted_sampler and class_weights is not None and not allow_loss_weights_with_sampler:
            print("\nDetected weighted sampler + class weights combination - "
                  "disabling class weights in loss to prevent over-weighting.")
            class_weights = None
        elif use_weighted_sampler and class_weights is not None and allow_loss_weights_with_sampler:
            print("\nEnabled combined weighted sampler + class weights (allow_loss_weights_with_sampler=true)")

        # Create loss function
        if loss_type == 'cross_entropy':
            label_smoothing = self.config.get('loss', {}).get('label_smoothing', 0.0)
            return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        elif loss_type == 'focal':
            gamma = self.config.get('loss', {}).get('gamma', 2.0)

            class FocalLoss(nn.Module):
                """Focal Loss implementation for imbalanced classification"""
                def __init__(self, alpha=None, gamma=2.0):
                    super().__init__()
                    self.alpha = alpha
                    self.gamma = gamma

                def forward(self, logits, targets):
                    log_probs = nn.functional.log_softmax(logits, dim=1)
                    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                    pt = log_pt.exp()

                    ce = -log_pt
                    if self.alpha is not None:
                        alpha_t = self.alpha.gather(0, targets)
                        ce = alpha_t * ce

                    loss = ((1 - pt) ** self.gamma) * ce
                    return loss.mean()

            return FocalLoss(alpha=class_weights, gamma=gamma)
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")

    def train_epoch(self) -> Dict[str, float]:
        """Train model for one epoch

        Returns:
            metrics: Dictionary containing training loss and accuracy
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        max_train_batches = self.config.get('max_train_batches_per_epoch')
        if max_train_batches is not None:
            max_train_batches = int(max_train_batches)
            if max_train_batches <= 0:
                raise ValueError("max_train_batches_per_epoch must be > 0")

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}",
            total=max_train_batches if max_train_batches is not None else None
        )

        for batch_idx, (images, labels) in enumerate(pbar):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Update metrics
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        # Calculate epoch metrics
        if total == 0:
            raise RuntimeError("No training samples were processed in train_epoch")

        avg_loss = total_loss / total
        accuracy = correct / total

        metrics = {
            'train_loss': avg_loss,
            'train_acc': accuracy
        }

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Evaluate model on validation dataset

        Returns:
            metrics: Dictionary containing validation metrics (loss, accuracy, F1)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # Collect predictions for F1 calculation
        all_preds = []
        all_labels = []

        max_val_batches = self.config.get('max_val_batches_per_epoch')
        if max_val_batches is not None:
            max_val_batches = int(max_val_batches)
            if max_val_batches <= 0:
                raise ValueError("max_val_batches_per_epoch must be > 0")

        val_pbar = tqdm(
            self.val_loader,
            desc="Validating",
            total=max_val_batches if max_val_batches is not None else None
        )

        for batch_idx, (images, labels) in enumerate(val_pbar):
            if max_val_batches is not None and batch_idx >= max_val_batches:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Update metrics
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Collect predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate validation metrics
        if total == 0:
            raise RuntimeError("No validation samples were processed in validate")

        avg_loss = total_loss / total
        accuracy = correct / total

        # Calculate F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        metrics = {
            'val_loss': avg_loss,
            'val_acc': accuracy,
            'val_f1': f1
        }

        # Calculate per-class and target class F1 scores
        class_names = list(self.config.get('data', {}).get('class_names', []))
        if class_names:
            per_class_f1 = f1_score(
                all_labels,
                all_preds,
                labels=list(range(len(class_names))),
                average=None,
                zero_division=0
            )

            valid_target_names = [
                class_name
                for class_name in self.target_class_names
                if class_name in class_names
            ]
            
            if valid_target_names:
                class_to_idx = {name: idx for idx, name in enumerate(class_names)}
                target_indices = [class_to_idx[name] for name in valid_target_names]
                target_f1 = float(np.mean(per_class_f1[target_indices]))
                metrics['val_target_f1'] = target_f1

                # Log per-target-class F1 scores
                for class_name, class_idx in zip(valid_target_names, target_indices):
                    metrics[f'val_f1_{class_name}'] = float(per_class_f1[class_idx])

        return metrics

    def train(self):
        """Complete training pipeline with validation and checkpointing"""
        print(f"\n{'='*60}")
        print(f"Starting training: {self.experiment_name}")
        print(f"{'='*60}\n")

        epochs = self.config.get('epochs', 50)
        patience = self.config.get('early_stopping_patience', 10)

        for epoch in range(self.start_epoch, epochs):
            self.current_epoch = epoch

            # Training phase
            train_metrics = self.train_epoch()

            # Validation phase
            val_metrics = self.validate()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Print epoch metrics
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Train Acc: {train_metrics['train_acc']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_acc']:.4f}, "
                  f"Val F1: {val_metrics['val_f1']:.4f}")
            
            if 'val_target_f1' in val_metrics:
                print(f"Val Target F1: {val_metrics['val_target_f1']:.4f}")

            # Log to WandB if enabled
            if self.wandb is not None:
                self.wandb.log({
                    **train_metrics,
                    **val_metrics,
                    'epoch': epoch,
                    'lr': self.optimizer.param_groups[0]['lr']
                })

            # Update training history
            current_metric = float(val_metrics[self.monitor_metric])
            self.history.append({
                'epoch': int(epoch),
                **{k: float(v) for k, v in train_metrics.items()},
                **{k: float(v) for k, v in val_metrics.items()},
                'lr': float(self.optimizer.param_groups[0]['lr'])
            })

            # Save best model checkpoint
            if self._is_better(current_metric):
                self.best_metric = current_metric
                self.best_val_acc = float(val_metrics['val_acc'])
                self.best_val_f1 = float(val_metrics['val_f1'])
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth')
                print(
                    f"✓ Saved best model "
                    f"({self.monitor_metric}: {self.best_metric:.4f}, "
                    f"Val Acc: {self.best_val_acc:.4f}, Val F1: {self.best_val_f1:.4f})"
                )
            else:
                self.patience_counter += 1

            # Early stopping check
            if self.patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

            # Periodic checkpoint saving
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

        # Training completion summary
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best {self.monitor_metric}: {self.best_metric:.4f}")
        print(f"Best Val Acc: {self.best_val_acc:.4f}")
        print(f"Best Val F1: {self.best_val_f1:.4f}")
        print(f"{'='*60}\n")

        # Save training history
        self.save_training_summary()

    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a saved checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if checkpoint.get('optimizer_state_dict') is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if self.scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore training state
        self.current_epoch = checkpoint.get('epoch', -1)
        self.start_epoch = self.current_epoch + 1
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)

        # Handle monitor metric configuration
        resume_use_checkpoint_monitor = self.config.get('resume_use_checkpoint_monitor', False)
        ckpt_monitor_metric = checkpoint.get('monitor_metric')
        ckpt_monitor_mode = checkpoint.get('monitor_mode')
        
        if resume_use_checkpoint_monitor and ckpt_monitor_metric is not None:
            self.monitor_metric = ckpt_monitor_metric
        if resume_use_checkpoint_monitor and ckpt_monitor_mode is not None:
            self.monitor_mode = ckpt_monitor_mode

        # Set best metric
        if self.monitor_metric == 'val_f1':
            default_best = self.best_val_f1
        elif self.monitor_metric == 'val_acc':
            default_best = self.best_val_acc
        else:
            default_best = -math.inf if self.monitor_mode == 'max' else math.inf

        if checkpoint.get('monitor_metric') == self.monitor_metric and 'best_metric' in checkpoint:
            self.best_metric = checkpoint['best_metric']
        else:
            self.best_metric = default_best

        # Restore training history
        self.history = checkpoint.get('history', [])
        self.patience_counter = 0

        print(f"Resumed from checkpoint: {checkpoint_path}")
        print(f"Resume start epoch: {self.start_epoch}")
        print(f"Best {self.monitor_metric} so far: {self.best_metric:.4f}")
        print(f"Best Val Acc so far: {self.best_val_acc:.4f}")
        print(f"Best Val F1 so far: {self.best_val_f1:.4f}")

        # Save baseline best_model in current experiment directory for complete archiving
        self.save_checkpoint('best_model.pth')

    def save_checkpoint(self, filename: str):
        """Save complete training checkpoint

        Checkpoint contains:
        - model_state_dict: Model weights
        - optimizer_state_dict: Optimizer state
        - scheduler_state_dict: LR scheduler state
        - epoch: Current training epoch
        - best_val_acc: Best validation accuracy
        - best_val_f1: Best validation F1 score
        - config: Training configuration
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.current_epoch,
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'best_metric': self.best_metric,
            'monitor_metric': self.monitor_metric,
            'monitor_mode': self.monitor_mode,
            'history': self.history,
            'config': self.config
        }

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)

    def save_training_summary(self):
        """Save training summary to JSON file for later analysis"""
        summary = {
            'experiment_name': self.experiment_name,
            'monitor_metric': self.monitor_metric,
            'monitor_mode': self.monitor_mode,
            'best_metric': float(self.best_metric),
            'best_val_acc': float(self.best_val_acc),
            'best_val_f1': float(self.best_val_f1),
            'total_epochs': self.current_epoch + 1,
            'history': self.history,
            'config': self.config
        }

        summary_path = self.checkpoint_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Saved training summary to {summary_path}")


def load_checkpoint(checkpoint_path: str, model: nn.Module) -> Dict:
    """Load model weights and training metadata from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load weights into

    Returns:
        checkpoint: Complete checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Best Val Acc: {checkpoint['best_val_acc']:.4f}")
    
    if 'best_val_f1' in checkpoint:
        print(f"Best Val F1: {checkpoint['best_val_f1']:.4f}")
    
    if 'monitor_metric' in checkpoint and 'best_metric' in checkpoint:
        print(
            f"Best {checkpoint['monitor_metric']}: "
            f"{checkpoint['best_metric']:.4f} ({checkpoint.get('monitor_mode', 'max')})"
        )

    return checkpoint


if __name__ == '__main__':
    # Test Trainer functionality
    from src.data.dataset import create_dataloaders

    # Create data loaders (assumes data in data/raw)
    try:
        train_loader, val_loader = create_dataloaders(
            data_root='data/raw',
            batch_size=8,
            num_workers=2,
            img_size=256
        )
    except Exception as e:
        print(f"Warning: Could not create dataloaders: {e}")
        print("Creating dummy dataloaders for testing...")

        # Create dummy datasets for testing
        from torch.utils.data import TensorDataset, DataLoader

        dummy_train = TensorDataset(
            torch.randn(100, 3, 256, 256),
            torch.randint(0, 4, (100,))
        )
        dummy_val = TensorDataset(
            torch.randn(20, 3, 256, 256),
            torch.randint(0, 4, (20,))
        )

        train_loader = DataLoader(dummy_train, batch_size=8, shuffle=True)
        val_loader = DataLoader(dummy_val, batch_size=8)

    # Create model
    model = create_resnet_model(
        backbone='resnet50',
        num_classes=4,
        pretrained=False
    )

    # Training configuration
    config = {
        'epochs': 2,
        'learning_rate': 1e-3,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'use_amp': True,
        'early_stopping_patience': 5,
    }

    # Create trainer instance
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        experiment_name='test',
        use_wandb=False
    )

    # Start training
    trainer.train()
