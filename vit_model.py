# VISION TRANSFORMER - FINE-TUNING MODEL

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTModel
from typing import Optional


# MODEL 1: COMPLETE FINE-TUNING
class ViTFineTuner(nn.Module):
    """
    Vision Transformer with complete fine-tuning.
    It uses a pre-trained model and modifies the last layer for the binary classification.
    """
    
    def __init__(self, model_name='google/vit-base-patch16-224', num_classes=2, freeze_backbone=False):
        """
        Args:
            model_name (str): Name of the pre-trained model from HuggingFace
            num_classes (int): Number of classes (2 for Real vs AI)
            freeze_backbone (bool): If True, it freezes weights of the backbone (only trainable classifier)
        """
        super(ViTFineTuner, self).__init__()
        
        # Load pre-trained model
        self.vit = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True 
        )
        
        # Optional: freeze the backbone
        if freeze_backbone:
            print("Backbone freezed - Only trainable classifier")
            for param in self.vit.vit.parameters():
                param.requires_grad = False
        else:
            print("Complete Fine-tuning - All layers trainable")
        
        # Counts trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits



# MODEL 2: FINE-TUNING WITH CUSTOM CLASSIFIER
class ViTWithCustomHead(nn.Module):
    """
    Vision Transformer with a custom classifier more sofisticated.
    Useful if we want to add dropout, layers, ecc.
    """
    
    def __init__(self, model_name='google/vit-base-patch16-224', num_classes=2, dropout=0.1):
        super(ViTWithCustomHead, self).__init__()
        
        # Load only the backbone (without classifier)
        self.vit = ViTModel.from_pretrained(model_name)
        
        # ViT backbone outpout size
        hidden_size = self.vit.config.hidden_size  # 768 for vit-base
        
        # Custom classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        print(f"Custom classifier created: {hidden_size} -> {hidden_size // 2} -> {num_classes}")
    
    def forward(self, pixel_values):
        # Extract features from backbone
        outputs = self.vit(pixel_values=pixel_values)
        
        # Use [CLS] token (first token) for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        logits = self.classifier(cls_output)
        return logits



# FUNCTION FACTORY
def create_model(
    model_type='fine_tune',
    model_name='google/vit-base-patch16-224',
    num_classes=2,
    freeze_backbone=False,
    dropout=0.1
):
    """
    Factory function to create the model.
    
    Args:
        model_type (str): 'fine_tune' or 'custom_head'
        model_name (str): Name of pre-trained model
        num_classes (int): Class number
        freeze_backbone (bool): Freeze backbone (only for fine_tune)
        dropout (float): Dropout rate (only for custom_head)
    
    Returns:
        model, optimizer, scheduler
    """
    
    print(f"\nModel Creation: {model_type}")
    print(f"Base Model: {model_name}")
    
    if model_type == 'fine_tune':
        model = ViTFineTuner(
            model_name=model_name,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone
        )
    elif model_type == 'custom_head':
        model = ViTWithCustomHead(
            model_name=model_name,
            num_classes=num_classes,
            dropout=dropout
        )
    else:
        raise ValueError(f"model_type needs to be 'fine_tune' or 'custom_head', received: {model_type}")
    
    return model