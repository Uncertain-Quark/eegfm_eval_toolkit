# CBraMod implementation
# https://arxiv.org/pdf/2412.07236
# From https://github.com/wjq-learning/CBraMod/blob/main/models/cbramod.py
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import MSELoss

from biodl.layers.patch_embeddings import PatchEmbedding
from biodl.layers.transformer_encoder import TransformerEncoder, TransformerEncoderLayer
from biodl.models.base import BaseModel

from peft import get_peft_model, LoraConfig, TaskType

def generate_mask(bz, ch_num, patch_num, mask_ratio):
    mask = torch.zeros((bz, ch_num, patch_num), dtype=torch.long)
    mask = mask.bernoulli_(mask_ratio)
    return mask

class PeftCompatWrapper(nn.Module):
    """
    Wraps a custom model to make it compatible with PEFT's expected input signature.
    PEFT expects 'input_ids', but CBraMod expects 'x'.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids=None, x=None, **kwargs):
        # PEFT treats the first positional arg as 'input_ids'.
        # We capture it and pass it to CBraMod as 'x'.
        inputs = x if x is not None else input_ids
        
        # We filter out kwargs that CBraMod doesn't accept to prevent errors
        # (CBraMod forward only accepts x, mask, labels)
        valid_args = {}
        if "mask" in kwargs:
            valid_args["mask"] = kwargs["mask"]
        if "labels" in kwargs:
            valid_args["labels"] = kwargs["labels"]
            
        return self.model(inputs, **valid_args)

class CBraMod(nn.Module):
    def __init__(
            self, in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, 
            seq_len=30, n_layer=12, nhead=8,
            sampling_rate=200, mask_ratio=0.5,
            **kwargs
        ):
        super().__init__()
        self.d_model = d_model
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, norm_first=True,
            activation=F.gelu
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)
        self.proj_out = nn.Sequential(
            # nn.Linear(d_model, d_model*2),
            # nn.GELU(),
            # nn.Linear(d_model*2, d_model),
            # nn.GELU(),
            nn.Linear(d_model, out_dim),
        )
        self.apply(_weights_init)

        self.loss_function = MSELoss(reduction='mean')
        self.mask_ratio = mask_ratio
    

    def forward(self, x, mask=None, labels=None):
        batch_size, num_channels, seq_len = x.shape
        time_segments = seq_len // self.d_model
        loss = None

        x = x.reshape(batch_size, num_channels, time_segments, self.d_model)
        # x = rearrange(x, 'b n s -> b n t p', t=time_segments, p=self.d_model)
        

        patch_emb = self.patch_embedding(x, mask)
        feats = self.encoder(patch_emb)

        out = self.proj_out(feats)

        if labels is not None:
            bz, ch_num, patch_num, patch_size = x.shape
            mask = generate_mask(
                bz, ch_num, patch_num, mask_ratio=self.mask_ratio
            )
            masked_x = x[mask == 1]
            masked_y = out[mask == 1]
            loss = self.loss_function(masked_y, masked_x)

        return {
            "logits": None, 
            "loss": loss, 
            "embeddings": out
            }

class CBraMod_Classification(BaseModel):
    def __init__(
            self,
            n_classes: int=None,
            d_model: int=None,
            pretrained_checkpoint: str=None,
            classifier_type: str="avgpooling_patch_reps",
            finetune_type: str="full_finetune",
            label_smoothing: float=0.1,
            dropout: float=0.1,
            n_seconds_input: int=None,
            n_channels: int=None,
            **kwargs
    ):
        super().__init__()
        self.loss_function = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        base_backbone = CBraMod(**kwargs, d_model=d_model)
        self.n_classes = n_classes
        self.finetune_type = finetune_type

        if pretrained_checkpoint is not None:
            if not os.path.exists(pretrained_checkpoint):
                pretrained_checkpoint = os.path.join(os.getenv("BIODL_PRETRAINED_CHECKPOINT_DIR", "./pretrain_checkpoints"), "cbramod", pretrained_checkpoint)
            print(f"Loading pretrained checkpoint from: {pretrained_checkpoint}")
            map_location = torch.device(f'cuda:0')
            base_backbone.load_state_dict(torch.load(pretrained_checkpoint, map_location=map_location))

        if finetune_type == "lora":
            print("Injecting LoRA adapters into the backbone...")

            self.backbone = PeftCompatWrapper(base_backbone)
            
            # Extract specific PEFT args from kwargs or set defaults
            lora_r = kwargs.get("lora_r", 8)
            lora_alpha = kwargs.get("lora_alpha", 16)
            lora_dropout = kwargs.get("lora_dropout", 0.1)
            
            target_modules = kwargs.get("target_modules", ["linear1", "linear2", "out_proj"]) 

            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                inference_mode=False, 
                r=lora_r, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
                target_modules=target_modules
            )
            
            self.backbone = get_peft_model(self.backbone, peft_config)
            self.backbone.print_trainable_parameters()
        else:
            self.backbone = base_backbone
        
        if classifier_type == "avgpooling_patch_reps":
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, n_classes),
            )
        elif classifier_type == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(int(n_channels * n_seconds_input * 200), n_classes),
            )
        elif classifier_type == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(int(n_channels * n_seconds_input * 200), 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(200, n_classes),
            )
        elif classifier_type == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(int(n_channels * n_seconds_input * 200), int(n_seconds_input * 200)),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(int(n_seconds_input * 200, 200)),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(200, n_classes),
            )
        
        if finetune_type == "linear_probe":
            print(f"Freezing parameters of Cbramod encoder")
            for p in self.backbone.parameters():
                p.requires_grad = False
        
        self.n_params = sum([p.numel() for p in self.parameters() if p.requires_grad])
        print(f"Total trainable parameters: {self.n_params/1e6:.3f}M")

    def train(self, mode=True):
        super().train(mode)

        if self.finetune_type == "linear_probe":
            self.backbone.eval()

    def forward(
            self,
            x,
            labels=None,
            return_embeddings=False,
            **kwargs
    ):
        feats = self.backbone(x)["embeddings"]
        logits = self.classifier(feats)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels)
        
        return {
            "logits": logits,
            "loss": loss,
            "embeddings": feats if return_embeddings else None
        }
    
    def get_optimizer_params(self, weight_decay: float):
        """
        Groups parameters for the optimizer to apply weight decay selectively.
        
        Logic:
        1. Classifier Head: NO weight decay (as requested).
        2. Biases & Norms: NO weight decay (standard Transformer best practice).
        3. LoRA Adapters: NO weight decay (preserves low-rank adaptation stability).
        4. Backbone Weights (Conv/Linear): Apply weight decay (if trainable).
        """
        decay = []
        no_decay = []

        for name, param in self.named_parameters():
            # 1. Skip frozen parameters 
            # (Crucial for 'linear_probe' and frozen parts of 'lora')
            if not param.requires_grad:
                continue
            
            # 2. Identify parameters to EXCLUDE from weight decay
            if (
                # "classifier" in name             # Your specific request
                "bias" in name                # Standard practice
                or "norm" in name                # LayerNorms in TransformerEncoder / GroupNorms
                or "lora" in name                # LoRA adapters (A/B matrices)
                or len(param.shape) == 1         # Catch-all for 1D params (scales/shifts)
            ):
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}
        ]
        

def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CBraMod(
        in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, 
        seq_len=30, n_layer=12, nhead=8
    ).to(device)
    a = torch.randn((8, 16, 10, 200)).cuda()
    b = model(a)
    print(a.shape, b.shape)