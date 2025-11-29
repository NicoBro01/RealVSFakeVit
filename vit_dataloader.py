# DATASET PREPARATION & DATALOADER FOR ViT FINE-TUNING
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from transformers import ViTImageProcessor



# CUSTOM DATASET CLASS
class RealVsAIDataset(Dataset):
    def __init__(self, root_dir, transform=None, processor=None):
        """
        Args:
            root_dir (str): Main directory (es: 'data/train')
            transform (callable, optional): Transformations to apply
            processor (ViTImageProcessor, optional): Preprocessor ViT
        """
        self.root_dir = root_dir
        self.transform = transform
        self.processor = processor
        
        # Load all images and labels
        self.images = []
        self.labels = []
        
        # 0 = REAL, 1 = FAKE/AI-Generated
        for label_name, label_id in [('REAL', 0), ('FAKE', 1)]:
            class_dir = os.path.join(root_dir, label_name)
            if not os.path.exists(class_dir):
                print(f"Directory not found: {class_dir}")
                continue
                
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(img_path)
                    self.labels.append(label_id)
        
        print(f"Loaded {len(self.images)} images from {root_dir}")
        print(f"   - REAL: {self.labels.count(0)}")
        print(f"   - FAKE: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transormations or preprocessor
        if self.processor:
            # Use ViTImageProcessor
            image = self.processor(images=image, return_tensors="pt")
            image = image['pixel_values'].squeeze(0)
        elif self.transform:
            image = self.transform(image)
        
        return image, label



# PREPROCESSING CONFIGURATION
def get_transforms(use_augmentation=True):
    """
    It returns transformations for train and validation
    
    Args:
        use_augmentation (bool): If True, it applies data augmentation to train set
    """
    
    # Base Transformations (for validation/test)
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ViT standard input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Transformations with DATA AUGMENTATION (for training)
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        return train_transform, base_transform
    
    return base_transform, base_transform



# CREAZIONE DATALOADER
def create_dataloaders(data_dir, batch_size=128, val_split=0.2, use_processor=True):
    """
    It creates DataLoader for training and validation
    
    Args:
        data_dir (str): Main directory of the dataset
        batch_size (int): Batch size
        val_split (float): Percentage of data for validation
        use_processor (bool): Use ViTImageProcessor instead of transforms
    
    Returns:
        train_loader, val_loader, num_classes
    """
    
    # Initializes preprocessor ViT
    processor = None
    if use_processor:
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        print("Using ViTImageProcessor")
    else:
        train_transform, val_transform = get_transforms(use_augmentation=True)
        print("Using custom transforms")
    
    # Load full dataset
    train_dir = os.path.join(data_dir, 'train')
    full_dataset = RealVsAIDataset(
        root_dir=train_dir,
        processor=processor if use_processor else None,
        transform=train_transform if not use_processor else None
    )
    
    # Split train/validation
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nSplit of data:")
    print(f"   - Training set: {train_size} images")
    print(f"   - Validation set: {val_size} images")

     
    NUM_WORKERS = 2      
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    PREFETCH_FACTOR = 2 
    
    # Create DataLoaders
    train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,           
    shuffle=True,
    num_workers=NUM_WORKERS,         
    pin_memory=PIN_MEMORY,           
    persistent_workers=PERSISTENT_WORKERS,  
    prefetch_factor=PREFETCH_FACTOR  
    )

    val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
    prefetch_factor=PREFETCH_FACTOR
    )
    
    return train_loader, val_loader, 2