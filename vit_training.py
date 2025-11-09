# COMPLETE TRAINING LOOP
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



# EARLY STOPPING
class EarlyStopping:
    """Early stopping to interrupt the training when validation loss doesn't improve."""
    
    def __init__(self, patience=7, min_delta=0, mode='min'):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement
            min_delta (float): Minimum change to consider it improvement
            mode (str): 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif self._is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:  # max
            return score > self.best_score + self.min_delta



# TRAINING EPOCH
def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Perform a training epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [TRAIN]')
    
    total_batches = len(train_loader) # Aggiunto per il calcolo della percentuale
    
    for batch_idx, (images, labels) in enumerate(pbar): # [1]
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Stats
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            print(f"Epoch {epoch} [TRAIN] - Iteration {batch_idx + 1}/{total_batches} | Current Loss: {loss.item():.4f}")

    # Compute metrics
    epoch_loss = running_loss / len(train_loader)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return epoch_loss, accuracy, precision, recall, f1



# VALIDATION EPOCH
def validate_epoch(model, val_loader, criterion, device, epoch):
    """Permorm a validation epoch."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [VAL]')
    total_batches = len(val_loader)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch} [VAL] - Iteration {batch_idx + 1}/{total_batches} | Current Loss: {loss.item():.4f}")
    
    # Compute metrics
    epoch_loss = running_loss / len(val_loader)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return epoch_loss, accuracy, precision, recall, f1, cm



# MAIN TRAINING LOOP
def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=20,
    learning_rate=2e-5,
    weight_decay=0.01,
    patience=5,
    save_path='best_model.pth',
    device='cuda'
):
    """
    Complete Training with early stopping and metrics.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        num_epochs: Max number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        patience: Patience for early stopping
        save_path: Path to save the best model
        device: Device (cuda/cpu)
    
    Returns:
        history: Dictionary with metrics for each epoch
    """
    
    print(f"\n{'='*60}")
    print(f"START TRAINING")
    print(f"{'='*60}")
    print(f"CONFIGURATION:")
    print(f"   - Epochs: {num_epochs}")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Weight decay: {weight_decay}")
    print(f"   - Early stopping patience: {patience}")
    print(f"   - Device: {device}")
    print(f"{'='*60}\n")
    
    # Setup
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: AdamW it's reccomended per Transformers
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, mode='min')
    
    # History to monitor metrics
    history = {
        'train_loss': [], 'train_acc': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    best_val_loss = float('inf')
    total_start_time = time.time()
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        
        # Training
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validation
        val_loss, val_acc, val_prec, val_rec, val_f1, cm = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_precision'].append(train_prec)
        history['train_recall'].append(train_rec)
        history['train_f1'].append(train_f1)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['val_f1'].append(val_f1)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_precision': val_prec,
            }, save_path)
            print(f"Model saved! Val Loss: {val_loss:.4f}")
        
        # Time epoch
        epoch_time = time.time() - epoch_start_time
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{num_epochs} - Time: {epoch_time:.2f}s")
        print(f"{'='*60}")
        print(f"TRAIN | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
              f"Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f}")
        print(f"VAL   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | "
              f"Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")
        print(f"{'='*60}\n")
        
        # Confusion matrix
        print("Confusion Matrix (Validation):")
        print(cm)
        print()
        
        # Early stopping check
        if early_stopping(val_loss, epoch):
            print(f"⏹Early stopping activated! Best epoch: {early_stopping.best_epoch}")
            break
    
    # Training completed
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved here: {save_path}")
    print(f"{'='*60}\n")
    
    return history


# METRICS PLOTTING
def plot_training_history(history, save_path='training_curves.png'):
    """Plot training and validation metrics over epochs."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = [
        ('loss', 'Loss'),
        ('acc', 'Accuracy'),
        ('precision', 'Precision'),
        ('f1', 'F1-Score')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        ax.plot(history[f'train_{metric}'], label='Train', marker='o')
        ax.plot(history[f'val_{metric}'], label='Validation', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'{title} over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Charts saved here: {save_path}")
    plt.show()