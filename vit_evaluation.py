# EVALUATION E TESTING
import torch
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# LOADING MODEL
def load_model(model, checkpoint_path, device='cuda'):
    """
    Load model weights from the checkpoint.
    
    Args:
        model: Model to load
        checkpoint_path: Path to the checkpoint
        device: Device (cuda/cpu)
    
    Returns:
        model: Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from: {checkpoint_path}")
    print(f"Epoch from the checkpoint: {checkpoint.get('epoch', 'N/A')}")
    print(f"Val Accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
    print(f"Val Precision: {checkpoint.get('val_precision', 'N/A'):.4f}")
    
    return model



# PREDIZIONE SUL TEST SET
def predict(model, test_loader, device='cuda'):
    """
    Perform predictions on the test set.
    
    Returns:
        all_preds: Array of predictions
        all_labels: Array of true label
        all_probs: Array of probabilities
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nPerforming predictions on the test set...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)



# COMPUTE METRICS
def compute_metrics(y_true, y_pred, y_probs, class_names=['REAL', 'FAKE']):
    """
    Compute and print all evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predictions
        y_probs: Probabilities for each class
        class_names: Names of the classes
    
    Returns:
        metrics_dict: Dictionary with all metrics 
    """
    
    print(f"\n{'='*60}")
    print("EVALUATION METRICS")
    print(f"{'='*60}\n")
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              REAL    FAKE")
    print(f"Actual REAL   {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"       FAKE   {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    # Accuracy for each class
    accuracy_real = cm[0,0] / (cm[0,0] + cm[0,1])
    accuracy_fake = cm[1,1] / (cm[1,0] + cm[1,1])
    overall_accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    
    print(f"\nAccuracy for each class:")
    print(f"   - REAL: {accuracy_real:.4f} ({accuracy_real*100:.2f}%)")
    print(f"   - FAKE: {accuracy_fake:.4f} ({accuracy_fake*100:.2f}%)")
    print(f"   - Overall: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    # Precision, Recall, F1 for each class
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    print(f"\nMetrics for each class:")
    for i, class_name in enumerate(class_names):
        print(f"\n{class_name}:")
        print(f"   - Precision: {precision[i]:.4f}")
        print(f"   - Recall: {recall[i]:.4f}")
        print(f"   - F1-Score: {f1[i]:.4f}")
    
    # ROC AUC (for FAKE class)
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    print(f"\nROC AUC Score: {roc_auc:.4f}")
    
    print(f"\n{'='*60}\n")
    
    # Create metrics dictionary
    metrics_dict = {
        'accuracy': overall_accuracy,
        'accuracy_real': accuracy_real,
        'accuracy_fake': accuracy_fake,
        'precision_real': precision[0],
        'precision_fake': precision[1],
        'recall_real': recall[0],
        'recall_fake': recall[1],
        'f1_real': f1[0],
        'f1_fake': f1[1],
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
    
    return metrics_dict



# VISUALIZATIONS
def plot_confusion_matrix(cm, class_names=['REAL', 'FAKE'], save_path='confusion_matrix.png'):
    """Visualizes confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved here: {save_path}")
    plt.show()

def plot_roc_curve(y_true, y_probs, save_path='roc_curve.png'):
    """Visualizes ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved here: {save_path}")
    plt.show()

def plot_precision_recall_curve(y_true, y_probs, save_path='pr_curve.png'):
    """Visualizes Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"PR curve saved here: {save_path}")
    plt.show()

def visualize_predictions(model, test_loader, device, num_images=16, save_path='predictions.png'):
    """Visualizes examples of predictions."""
    model.eval()
    
    # Obtain a batch
    images, labels = next(iter(test_loader))
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    
    # Convert for visualization
    images = images.cpu()
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    probs = probs.cpu().numpy()
    
    # Inverse normalization (if normalization was used)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    class_names = ['REAL', 'FAKE']
    
    for i in range(min(num_images, len(images))):
        img = images[i].permute(1, 2, 0).numpy()
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        confidence = probs[i][preds[i]]
        
        color = 'green' if labels[i] == preds[i] else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.2f})',
                         color=color, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualizations saved here: {save_path}")
    plt.show()



# COMPLETE FUNCTION OF EVALUATION
def evaluate_model(model, test_loader, device='cuda', save_dir='results/'):
    """
    It performs a complete evaluation of the model.
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for the test set
        device: Device (cuda/cpu)
        save_dir: Directory to save results
    """
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("START EVALUATION")
    print(f"{'='*60}\n")
    
    # Predictions
    y_pred, y_true, y_probs = predict(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_probs)
    
    # Visualizations
    print("\Generating visualizations...")
    plot_confusion_matrix(metrics['confusion_matrix'], 
                         save_path=os.path.join(save_dir, 'confusion_matrix.png'))
    plot_roc_curve(y_true, y_probs, 
                   save_path=os.path.join(save_dir, 'roc_curve.png'))
    plot_precision_recall_curve(y_true, y_probs,
                               save_path=os.path.join(save_dir, 'pr_curve.png'))
    visualize_predictions(model, test_loader, device,
                         save_path=os.path.join(save_dir, 'predictions.png'))
    
    # Save metrics on a file
    import json
    metrics_serializable = {k: float(v) if isinstance(v, np.number) else v.tolist() if isinstance(v, np.ndarray) else v 
                           for k, v in metrics.items()}
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_serializable, f, indent=4)
    
    print(f"\nEvaluation completed! Results saved here: {save_dir}")
    
    return metrics