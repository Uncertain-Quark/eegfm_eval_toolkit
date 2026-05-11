# EEGNet Implementation
# https://arxiv.org/abs/1611.08024

import torch
import torch.nn as nn
import torch.nn.functional as F

from biodl.models.base import BaseModel

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        # This ensures the weights are clipped to max_norm BEFORE the forward pass
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)

class EEGNet(BaseModel):

    def __init__(
            self,
            F1: int=8,
            D: int=2, 
            F2: int=None,
            sampling_rate: int=128, 
            n_seconds_input: int=5, 
            n_channels: int=None, 
            n_classes: int=None, 
            mode: str="between",
            seq2seq: bool=False,
            pretrained_checkpoint: str=None,
            conv_constraint: bool=False,
            **kwargs
            ):

        super(EEGNet, self).__init__()

        self.seq2seq = seq2seq
        self.p = 0.25 if mode == "between" else 0.5
        self.n_classes = n_classes
        self.n_timesteps = int(sampling_rate * n_seconds_input)
        self.n_channels = n_channels
        self.F1 = F1
        self.D = D
        self.F2 = self.F1 * self.D if F2 is None else F2

        # --- Block 1: Temporal & Depthwise Conv ---
        self.conv_temp = nn.Conv2d(in_channels=1, out_channels=self.F1,
                                   kernel_size=(1, sampling_rate//2), padding="same")
        self.batch_norm_temp = nn.BatchNorm2d(num_features=self.F1)

        if conv_constraint:
            print(f"Setting max norm of depth convolution!")
            self.conv_depthwise = Conv2dWithConstraint(F1, F1 * D, (n_channels, 1), 
                                                   groups=F1, padding="valid", bias=False, 
                                                   max_norm=1.0)
        else:
            self.conv_depthwise = nn.Conv2d(in_channels=self.F1, out_channels=self.F1*self.D,
                                            kernel_size=(self.n_channels, 1), padding="valid", groups=self.F1)
        
        self.batch_norm_depthwise = nn.BatchNorm2d(num_features=self.F1*self.D)
        self.activation = nn.ELU()
        self.avgpool_depthwise = nn.AvgPool2d(kernel_size=(1, 4), padding=(0, 0))
        self.dropout = nn.Dropout(p=self.p)

        # --- Block 2: Separable Conv ---
        self.conv_seperable = nn.Conv2d(in_channels=self.F1*self.D, out_channels=self.F1*self.D,
                                        kernel_size=(1, sampling_rate//8), padding="same", groups=self.F1*self.D)
        self.conv_pointwise = nn.Conv2d(in_channels=self.F1*self.D, out_channels=self.F2,
                                        kernel_size=1, padding="same")
        self.batch_norm_seperable = nn.BatchNorm2d(num_features=self.F2)
        self.avgpool_seperable = nn.AvgPool2d(kernel_size=(1, 8), padding=(0, 0))

        # --- Output Heads ---
        if self.seq2seq:
            # 1x1 Conv to project channels to n_classes (retaining time dim)
            self.projection = nn.Conv2d(in_channels=self.F2, out_channels=n_classes, kernel_size=1)
        else:
            # Standard Flatten + Linear
            self.flatten = nn.Flatten()
            # Note: The input size calculation relies on the exact pooling factors (4 * 8 = 32)
            self.fc = nn.Linear(in_features=self.F2*(self.n_timesteps//32), out_features=n_classes)
        
        if pretrained_checkpoint is not None:
            # load the checkpoint for the parameters that match shape
            checkpoint = torch.load(pretrained_checkpoint, map_location='cpu')
            
            # Unwrap state_dict if it is nested under a key (e.g., 'state_dict' or 'model')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                pretrained_dict = checkpoint['model_state_dict']
            else:
                pretrained_dict = checkpoint
            
            # Get current model state
            model_dict = self.state_dict()
            
            # Filter out unnecessary keys or keys with shape mismatches
            # This is critical if n_classes or input lengths differ from the pretrained model
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                               if k in model_dict and v.shape == model_dict[k].shape 
                               and not k.startswith('fc') 
                               and not k.startswith('projection')}
            
            # Overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            
            # Load the new state dict
            self.load_state_dict(model_dict)
            print(f"Loaded {len(pretrained_dict)} layers from {pretrained_checkpoint}")


    def forward(self, x, labels=None, return_embeddings: bool=False, **kwargs):
        loss = None
        
        # Input shape: (Batch, Channels, Time)
        original_length = x.shape[2]

        x = torch.unsqueeze(x, 1) # (Batch, 1, Channels, Time)

        # Block 1
        x = self.conv_temp(x)
        x = self.batch_norm_temp(x)
        x = self.conv_depthwise(x)
        x = self.batch_norm_depthwise(x)
        x = self.activation(x)
        x = self.avgpool_depthwise(x) # Time becomes T/4
        x = self.dropout(x)

        # Block 2
        x = self.conv_seperable(x)
        x = self.conv_pointwise(x)
        x = self.batch_norm_seperable(x)
        x = self.activation(x)
        x = self.avgpool_seperable(x) # Time becomes T/32
        x = self.dropout(x)

        if self.seq2seq:
            # x is currently (Batch, F2, 1, T/32)

            # 1. Project to classes
            logits = self.projection(x) # (Batch, n_classes, 1, T/32)

            # 2. Upsample back to original time length
            logits = F.interpolate(logits, size=(1, original_length), mode='bilinear', align_corners=False)

            # 3. Remove the singleton height dimension
            logits = logits.squeeze(2) # (Batch, n_classes, Time)

        else:
            x = self.flatten(x)
            logits = self.fc(x)

        if labels is not None:
            loss = self.loss_function(logits, labels)
        
        return {
                "logits": logits, 
                "loss": loss, 
                "embeddings": None if not return_embeddings else x
                }

if __name__ == "__main__":
    
    n_channels = 22
    n_seconds_input = 3
    sampling_rate = 250
    
    model = EEGNet(n_channels=n_channels, sampling_rate=sampling_rate, n_seconds_input=n_seconds_input, n_classes=4, F1=16, D=4)
    inputs = torch.randn(8, n_channels, int(n_seconds_input * sampling_rate))
    
    print(f"="*50)
    print(f"Shape of inputs: {inputs.shape}")
    outputs = model(inputs)
    print(f"="*50)
    print(f"Shape of output logits: {outputs['logits'].shape}")

    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Num params: {n_params}")


