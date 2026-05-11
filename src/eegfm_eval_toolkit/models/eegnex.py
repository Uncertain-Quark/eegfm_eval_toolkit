import torch
import torch.nn as nn
import torch.nn.functional as F

from biodl.models.base import BaseModel

# --- 1. Helper: Constrained Convolution ---
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)

# --- 2. Main EEGNeX Class ---
class EEGNeX(BaseModel):
    def __init__(
            self,
            n_classes: int,
            n_channels: int,
            n_seconds_input: int,
            sampling_rate: int,
            f1: int = 8,   # Temporal filters (Layer 1)
            f2: int = 32,  # Temporal filters (Layer 2) - This is the "Expanded" temporal bank
            d: int = 2,    # Depth multiplier
            mode: str = "between",
            pretrained_checkpoint: str = None,
            **kwargs
            ):
        super(EEGNeX, self).__init__()

        # --- Dynamic Kernel Sizes ---
        # Based on original paper usage of 250Hz:
        # Kernel 64 -> sr // 4
        # Kernel 16 -> sr // 16
        k1 = sampling_rate // 4 
        k2 = sampling_rate // 16 
        
        # Calculate Dropout based on mode (same logic as EEGNet)
        self.p = 0.25 if mode == "between" else 0.5 

        # --- Block 1: Double Temporal Convolution ---
        # Layer 1: Initial temporal filter
        self.conv_temp_1 = nn.Conv2d(1, f1, (1, k1), padding="same", bias=False)
        self.bn_temp_1 = nn.BatchNorm2d(f1)
        
        # Layer 2: Expanded temporal filter (The "NeX" addition)
        self.conv_temp_2 = nn.Conv2d(f1, f2, (1, k1), padding="same", bias=False)
        self.bn_temp_2 = nn.BatchNorm2d(f2)
        
        self.activation = nn.ELU()

        # --- Block 2: Spatial (Depthwise) Convolution ---
        # Note: Input channels = f2 (32), Output = f2 * d (64)
        self.conv_depthwise = Conv2dWithConstraint(
            f2, f2 * d, (n_channels, 1), 
            groups=f2, padding="valid", bias=False, max_norm=1.0
        )
        self.bn_depthwise = nn.BatchNorm2d(f2 * d)
        self.avgpool_1 = nn.AvgPool2d((1, 4))
        self.dropout_1 = nn.Dropout(p=self.p)

        # --- Block 3: Dilated Temporal Stack ---
        # Layer 4 in Keras code: Dilation = 2
        # Input: 64 channels -> Output: 32 channels (Reduction)
        self.conv_dilated_1 = nn.Conv2d(
            f2 * d, f2, (1, k2), 
            dilation=(1, 2), padding="same", bias=False
        )
        self.bn_dilated_1 = nn.BatchNorm2d(f2)

        # Layer 5 in Keras code: Dilation = 4
        # Input: 32 channels -> Output: 8 channels (Final bottleneck)
        self.conv_dilated_2 = nn.Conv2d(
            f2, f1, (1, k2), 
            dilation=(1, 4), padding="same", bias=False
        )
        self.bn_dilated_2 = nn.BatchNorm2d(f1)
        self.avgpool_2 = nn.AvgPool2d((1, 4)) # Note: Keras code used (1,4) here too
        self.dropout_2 = nn.Dropout(p=self.p)

        # --- Classification Head ---
        # Dynamic flattening size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, int(sampling_rate * n_seconds_input))
            x = self.activation(self.bn_temp_1(self.conv_temp_1(dummy)))
            x = self.bn_temp_2(self.conv_temp_2(x)) # No activation here in original code
            x = self.activation(self.bn_depthwise(self.conv_depthwise(x)))
            x = self.dropout_1(self.avgpool_1(x))
            
            x = self.bn_dilated_1(self.conv_dilated_1(x)) # No activation
            x = self.activation(self.bn_dilated_2(self.conv_dilated_2(x)))
            x = self.dropout_2(self.avgpool_2(x))
            
            self.flatten_size = x.shape[1] * x.shape[2] * x.shape[3]

        self.flatten = nn.Flatten()
        # Original paper constrains the final dense layer too (max_norm=0.25)
        # However, PyTorch Linear doesn't support max_norm natively.
        # We handle this in the forward pass or use a custom Linear class.
        self.fc = nn.Linear(self.flatten_size, n_classes)

    def forward(self, x, labels=None, return_embeddings=False, **kwargs):
        # Input: (Batch, Channels, Time) -> (Batch, 1, Channels, Time)
        x = x.unsqueeze(1)

        # 1. Double Temporal
        x = self.conv_temp_1(x)
        x = self.activation(self.bn_temp_1(x))
        
        x = self.conv_temp_2(x)
        x = self.bn_temp_2(x) # Note: Often no activation between these stacked layers in NeX

        # 2. Depthwise Spatial
        x = self.conv_depthwise(x)
        x = self.activation(self.bn_depthwise(x))
        x = self.avgpool_1(x)
        x = self.dropout_1(x)

        # 3. Dilated Stack
        x = self.conv_dilated_1(x)
        x = self.bn_dilated_1(x) # No activation in middle of dilated stack usually
        
        x = self.conv_dilated_2(x)
        x = self.activation(self.bn_dilated_2(x))
        x = self.avgpool_2(x)
        x = self.dropout_2(x)

        # 4. Classifier
        embeddings = self.flatten(x)
        
        # Apply MaxNorm constraint to Linear layer weights manually before use (0.25 is standard)
        if self.training:
            with torch.no_grad():
                self.fc.weight.data = torch.renorm(self.fc.weight.data, p=2, dim=0, maxnorm=0.25)

        logits = self.fc(embeddings)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels)
            
        return {
            "logits": logits, 
            "loss": loss,
            "embeddings": embeddings if return_embeddings else None
        }
    
if __name__ == "__main__":
    n_classes = 4
    sampling_rate = 160
    n_channels = 64
    f1, f2 = 8, 32
    n_seconds_input = 4
    
    model = EEGNeX(
        n_classes=n_classes,
        n_channels=n_channels,
        sampling_rate=sampling_rate,
        n_seconds_input=n_seconds_input,
        f1=f1, f2=f2
    )

    n_params = sum([p.numel() for p in model.parameters() for p in model.parameters() if p.requires_grad])


    print(f"Number of parameters: {n_params}")