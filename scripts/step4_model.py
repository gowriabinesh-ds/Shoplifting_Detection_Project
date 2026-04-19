"""
STEP 4: MODEL DEFINITION
==========================
Uses R3D-18 pretrained on Kinetics-400 (400 human actions).
We remove the original 400-class head and replace with a 2-class head
(normal vs shoplifting).
"""

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class ShopliftingModel(nn.Module):
    """
    R3D-18 fine-tuned for shoplifting detection.
    Input:  (batch, 3, 16, 112, 112)  — batch of video clips
    Output: (batch, 2)                — logits for [normal, shoplifting]
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # Load R3D-18 with Kinetics-400 pretrained weights
        base = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)

        # Freeze the early layers — they already learned good motion features
        # We only train the later layers and the new head
        for param in base.stem.parameters():
            param.requires_grad = False          # Freeze stem (very early layer)
        for param in base.layer1.parameters():
            param.requires_grad = False          # Freeze layer1

        # Keep all other layers trainable
        self.stem   = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        # Replace 400-class Kinetics head with 2-class head
        in_features = base.fc.in_features        # 512 for R3D-18
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),                   # Helps prevent overfitting
            nn.Linear(in_features, num_classes), # Our 2-class output
        )

    def forward(self, x):
        # x shape: (batch, 3, T, H, W)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)     # Flatten to (batch, 512)
        x = self.fc(x)       # Output (batch, 2)
        return x


def get_model(num_classes: int = 2) -> nn.Module:
    print("Loading R3D-18 with Kinetics-400 pretrained weights...")
    model = ShopliftingModel(num_classes=num_classes)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"✅ Model ready — {trainable:,} trainable / {total:,} total parameters")
    return model


if __name__ == "__main__":
    model = get_model()
    # Test with a dummy input — 2 clips, 3 channels, 16 frames, 112x112
    dummy  = torch.randn(2, 3, 16, 112, 112)
    output = model(dummy)
    print(f"✅ Output shape: {output.shape}")   # Should be (2, 2)
