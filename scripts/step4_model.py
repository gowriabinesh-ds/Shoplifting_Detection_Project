"""
STEP 4: MODEL DEFINITION
==========================
Uses R3D-18 pretrained on Kinetics-400 (400 human actions).
We remove the original 400-class head and replace with a 2-class head
(normal vs shoplifting).
"""

import torch                      # PyTorch main library
import torch.nn as nn             # Neural network modules
from torchvision.models.video import r3d_18, R3D_18_Weights   # Pretrained 3D CNN model


class ShopliftingModel(nn.Module):   # Define custom model class
    """
    R3D-18 fine-tuned for shoplifting detection.
    Input:  (batch, 3, 16, 112, 112)  — batch of video clips
    Output: (batch, 2)                — logits for [normal, shoplifting]
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()                      # Initialize parent class

        # Load pretrained R3D-18 model trained on 400 human actions
        base = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)

        # Freeze the early layers (keep learned features, don’t retrain)
        # We only train the later layers and the new head
        for param in base.stem.parameters():
            param.requires_grad = False          # Freeze stem (very early layer)
        for param in base.layer1.parameters():
            param.requires_grad = False          # Freeze layer1 (freeze first block of layers)

        # Keep all other layers trainable (fine-tune deeper features)
        self.stem   = base.stem           # Early feature extraction layer
        self.layer1 = base.layer1         # First convolution block
        self.layer2 = base.layer2         # Second convolution block
        self.layer3 = base.layer3         # Third convolution block
        self.layer4 = base.layer4         # Fourth convolution block
        self.avgpool = base.avgpool       # Global average pooling layer

        # Replace original 400-class classifier with 2-class classifier
        in_features = base.fc.in_features        # 512 for R3D-18
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),                   # Reduce overfitting
            nn.Linear(in_features, num_classes), # Output 2 classes
        )

    def forward(self, x):               # Forward pass: defines how data flows through model
        # x shape: (batch, 3, T, H, W)
        x = self.stem(x)        # Extract basic motion features
        x = self.layer1(x)      # Learn simple patterns
        x = self.layer2(x)      # Learn more complex patterns
        x = self.layer3(x)      # Learn deeper motion patterns
        x = self.layer4(x)      # Learn high-level features
        x = self.avgpool(x)     # Reduce spatial dimensions
        x = x.flatten(1)        # Convert to 1D vector (batch, 512)
        x = self.fc(x)          # Final classification (2 classes)
        return x                # Return predictions


def get_model(num_classes: int = 2) -> nn.Module:
    print("Loading R3D-18 with Kinetics-400 pretrained weights...")    # Inform user
    model = ShopliftingModel(num_classes=num_classes)                  # Create model
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)   # Count trainable parameters
    total     = sum(p.numel() for p in model.parameters())                      # Count total parameters
    print(f"✅ Model ready — {trainable:,} trainable / {total:,} total parameters")   # Show model size
    return model


if __name__ == "__main__":
    model = get_model()    # Create model instance
    
    # Test with a dummy input — 2 clips, 3 channels, 16 frames, 112x112
    dummy  = torch.randn(2, 3, 16, 112, 112)
    output = model(dummy)
    print(f"✅ Output shape: {output.shape}")   # Should be (2, 2 → 2 samples, 2 classes)
