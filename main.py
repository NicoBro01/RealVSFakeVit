# MAIN SCRIPT - Real vs AI Image Classification
# Vision Transformer Fine-Tuning Project
import torch
import os
import argparse
from pathlib import Path

from vit_dataloader import create_dataloaders
from vit_model import create_model
from vit_training import train_model, plot_training_history, plot_batch_history
from vit_evaluation import load_model, evaluate_model

torch.set_float32_matmul_precision('high')

# PARAMETERS CONFIGURATION
class Config:

    # Paths
    DATA_DIR = "./data/cifake"
    SAVE_DIR = "./results"
    MODEL_SAVE_PATH = "./results/best_model.pth"
    
    # Model
    MODEL_NAME = "google/vit-base-patch16-224"  # Options: vit-base, vit-large, deit-base
    MODEL_TYPE = "fine_tune"  # Options: 'fine_tune', 'custom_head'
    FREEZE_BACKBONE = True  # False for complete fine-tuning, True for training only the classifier head
    
    # Training
    BATCH_SIZE = 64  # Reduce to 16 if there are GPU problems
    NUM_EPOCHS = 2
    LEARNING_RATE = 2e-5  # Usual Learning rate for fine-tuning
    WEIGHT_DECAY = 0.01
    VAL_SPLIT = 0.2  # 20% for validation
    
    # Early Stopping
    PATIENCE = 5  # Number of epochs without improvement before stopping
    
    # Hardware
    NUM_WORKERS = 2  # Threads for DataLoader (to adapt based on CPU)
    
    # Random Seed
    SEED = 42

def setup_directories():
    """It creates necessary directory."""
    Path(Config.SAVE_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Directory created: {Config.SAVE_DIR}")

def check_dataset():
    """It verifies thet the datased exists."""
    train_dir = Path(Config.DATA_DIR) / "train"
    
    if not train_dir.exists():
        print(f"ERROR: Dataset not found in {train_dir}")
        print("\nInstructions:")
        print("1. Download the CIFAKE dataset:")
        print("   kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images")
        print("\n2. Extract the dataset in the correct structure:")
        print(f"   {Config.DATA_DIR}/")
        print("   ├── train/")
        print("   │   ├── REAL/")
        print("   │   └── FAKE/")
        print("   └── test/")
        print("       ├── REAL/")
        print("       └── FAKE/")
        return False
    
    # Count immages
    real_train = len(list((train_dir / "REAL").glob("*.png")))
    fake_train = len(list((train_dir / "FAKE").glob("*.png")))
    
    print(f"Dataset found!")
    print(f"   - REAL images: {real_train}")
    print(f"   - FAKE images: {fake_train}")
    
    return True



# FUNZIONE MAIN
def main():
    """Main function - perform the complete pipeline."""
    
    print("\n" + "="*60)
    print("VISION TRANSFORMER - REAL VS AI CLASSIFICATION")
    print("="*60 + "\n")
    
    # Setup
    setup_directories()
    
    # Verifies dataset
    if not check_dataset():
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    
    # PHASE 1: LOAD DATA
    print("\n" + "="*60)
    print("PHASE 1: LOAD DATA")
    print("="*60 + "\n")
    
    train_loader, val_loader, num_classes = create_dataloaders(
        data_dir=Config.DATA_DIR,
        batch_size=Config.BATCH_SIZE,
        val_split=Config.VAL_SPLIT,
        use_processor=False
    )
    
    
    # PHASE 2: MODEL CREATION
    print("\n" + "="*60)
    print("PHASE 2: MODEL CREATION")
    print("="*60 + "\n")
    
    model = create_model(
        model_type=Config.MODEL_TYPE,
        model_name=Config.MODEL_NAME,
        num_classes=num_classes,
        freeze_backbone=Config.FREEZE_BACKBONE
    )

    
    
    # PHASE 3: TRAINING
    print("\n" + "="*60)
    print("PHASE 3: TRAINING")
    print("="*60 + "\n")
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=Config.NUM_EPOCHS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        patience=Config.PATIENCE,
        save_path=Config.MODEL_SAVE_PATH,
        device=device
    )
    
    # Plot training curves
    plot_training_history(
        history, 
        save_path=os.path.join(Config.SAVE_DIR, 'epoch_training_curves.png')
    )

    plot_batch_history(
        history, 
        save_path=os.path.join(Config.SAVE_DIR, 'batch_training_curves.png')
    )

    

    # PHASE 4: EVALUATION
    print("\n" + "="*60)
    print("PHASE 4: EVALUATION")
    print("="*60 + "\n")
    
    # Load test set
    # If we have a separate test set, we load it:
    test_dir = os.path.join(Config.DATA_DIR, "test")
    if os.path.exists(test_dir):
        from vit_dataloader import RealVsAIDataset
        from torch.utils.data import DataLoader
        from transformers import ViTImageProcessor
        
        processor = ViTImageProcessor.from_pretrained(Config.MODEL_NAME)
        test_dataset = RealVsAIDataset(test_dir, processor=processor)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                                shuffle=False, num_workers=Config.NUM_WORKERS)
    else:
        # Otherwise uses validation set as test
        test_loader = val_loader
    
    # Load best model
    model = create_model(
        model_type=Config.MODEL_TYPE,
        model_name=Config.MODEL_NAME,
        num_classes=num_classes
    )
    model = load_model(model, Config.MODEL_SAVE_PATH, device)
    
    # Evaluate
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        save_dir=Config.SAVE_DIR
    )
    
    # Verify goal
    print("\n" + "="*60)
    print("VERIFICA OBIETTIVO")
    print("="*60)
    
    target_precision = 0.80
    fake_precision = metrics['precision_fake']
    
    if fake_precision >= target_precision:
        print(f"\GOAL ACHIEVED!")
        print(f"   Precision FAKE: {fake_precision:.4f} (>= {target_precision})")
    else:
        print(f"\GOAL NOT ACHIEVED!")
        print(f"   Precision FAKE: {fake_precision:.4f} (< {target_precision})")
        print(f"   Servono ancora {(target_precision - fake_precision)*100:.2f} punti percentuali")
    
    print("\nFinal Metrics:")
    print(f"   - Accuracy: {metrics['accuracy']:.4f}")
    print(f"   - Precision REAL: {metrics['precision_real']:.4f}")
    print(f"   - Precision FAKE: {metrics['precision_fake']:.4f}")
    print(f"   - Recall REAL: {metrics['recall_real']:.4f}")
    print(f"   - Recall FAKE: {metrics['recall_fake']:.4f}")
    print(f"   - F1 REAL: {metrics['f1_real']:.4f}")
    print(f"   - F1 FAKE: {metrics['f1_fake']:.4f}")
    print(f"   - ROC AUC: {metrics['roc_auc']:.4f}")
    

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60 + "\n")
    
    print(f"📁 Results saved in: {Config.SAVE_DIR}")
    print("   - best_model.pth: Trained model")
    print("   - training_curves.png: Training Curves")
    print("   - confusion_matrix.png: Confusion matrix")
    print("   - roc_curve.png: ROC curve")
    print("   - pr_curve.png: Precision-Recall curve")
    print("   - predictions.png: Prediction examples")
    print("   - metrics.json: Metrics in JSON format")



# COMMAND LINE INTERFACE
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vision Transformer Fine-Tuning')
    
    parser.add_argument('--data_dir', type=str, default=Config.DATA_DIR,
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--model', type=str, default=Config.MODEL_NAME,
                       help='Pre-trained model name')
    
    args = parser.parse_args()
    
    # Updates config with args
    Config.DATA_DIR = args.data_dir
    Config.BATCH_SIZE = args.batch_size
    Config.NUM_EPOCHS = args.epochs
    Config.LEARNING_RATE = args.lr
    Config.MODEL_NAME = args.model
    
    # Exacute main function
    main()