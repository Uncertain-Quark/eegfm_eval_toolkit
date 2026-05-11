# Implementation of CSBrain, Code adapted from: https://github.com/yuchen2199/CSBrain 
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
from biodl.layers.csbrain_transformerlayer import *
from biodl.layers.csbrain_transformer import *
from biodl.models.base import BaseModel
from collections import Counter
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import uuid

from einops.layers.torch import Rearrange

from peft import get_peft_model, LoraConfig, TaskType

BRAIN_REGION_ENCODING = {
    "frontal": 0,
    "parietal": 1,
    "temporal": 2,
    "occipetal": 3,
    "central": 4
}

FULL_TOPOLOGY = {
    0: [
        # Frontal Pole
        'FP1', 'FPZ', 'FP2', 
        # Anterior Frontal
        'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
        # Frontal
        'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
        'FPZ-CZ',
        # Fronto-Temporal (Placed in Frontal as per your snippet)
        'FT9', 'FT7', 'FT8', 'FT10',
        # Fronto-Central
        'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6'
    ],

    4: [
        # Central
        'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
        # Centro-Parietal
        'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6'
    ],

    2: [
        # Temporal
        'T9', 'T7', 'T8', 'T10',
        # Older Nomenclature / Extras often found in datasets
        'T3', 'T5', 'T4', 'T6', 
        # Temporo-Parietal (Placed in Temporal as per your snippet)
        'TP9', 'TP7', 'TP8', 'TP10'
    ],

    1: [
        # Parietal
        'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10'
    ],

    3: [
        # Parieto-Occipital
        'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ',
        'PZ-OZ',
        'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
        # Occipital
        'O1', 'OZ', 'O2', 'O9', 'O10',
        # Landmarks
        'IZ'
    ]
}

def get_sorted_indices(channel_names, brain_regions):
    """
    Generates the sorted_indices required by CSBrain.
    
    Args:
        channel_names (list of str): Your dataset's channel names (e.g. ['Fz', 'C3'])
        brain_regions (list of int): The region ID for each channel (e.g. [0, 4])
        
    Returns:
        list of int: Indices to permute the input tensor.
    """
    
    # 1. Pre-compute rank maps for O(1) lookup
    # Structure: { region_id: { 'CHANNEL_NAME': rank_index } }
    topology_ranks = {}
    for r_id, ch_list in FULL_TOPOLOGY.items():
        topology_ranks[r_id] = {name.upper(): i for i, name in enumerate(ch_list)}

    # 2. Build a list of (original_index, sort_key) tuples
    # We need to preserve the original index to return it later
    combined = []
    
    for idx, (name, region) in enumerate(zip(channel_names, brain_regions)):
        name_upper = name.upper()
        
        # Validation: Ensure the region exists in our topology map
        if region not in topology_ranks:
            # If you pass a region ID not in 0-4, we put it at the very end
            rank = float('inf')
        else:
            # Look up the channel's rank within its specific region list
            # If channel is unknown/typo, put it at the end of that region (float('inf'))
            rank = topology_ranks[region].get(name_upper, float('inf'))
        
        combined.append({
            'original_index': idx,
            'region': region,
            'sub_rank': rank,
            'name': name
        })

    # 3. THE SORTING LOGIC
    # Priority 1: Region ID (Ascending) -> Groups 0s, then 1s, etc.
    # Priority 2: Sub-Rank (Ascending) -> Orders Fp1 before Fz
    combined.sort(key=lambda x: (x['region'], x['sub_rank']))

    # 4. Extract indices
    sorted_indices = [item['original_index'] for item in combined]
    
    return sorted_indices

# def get_brain_regions(channel_names):
#     """Given a list of channel names, return the corresponding brain region encoding"""
#     brain_regions = []

#     for channel_name in channel_names:
#         if channel_name.upper().startswith("A"):
#             brain_regions.append(BRAIN_REGION_ENCODING["frontal"])
#         elif "O" in channel_name.upper() or channel_name.upper().startswith("I"):
#             brain_regions.append(BRAIN_REGION_ENCODING["occipetal"])
#         elif channel_name.upper().startswith("P"):
#             brain_regions.append(BRAIN_REGION_ENCODING["parietal"])
#         elif "FT" in channel_name.upper():
#             brain_regions.append(BRAIN_REGION_ENCODING["temporal"])
#         elif channel_name.startswith("T"):
#             brain_regions.append(BRAIN_REGION_ENCODING["temporal"])
#         elif channel_name.startswith("F"):
#             brain_regions.append(BRAIN_REGION_ENCODING["frontal"])
#         elif channel_name.startswith("C"):
#             brain_regions.append(BRAIN_REGION_ENCODING["central"])
#     return brain_regions

def get_brain_regions(channel_names):
    """
    Robustly maps the 64-channel list to brain regions.
    Prioritizes specific 2-letter prefixes before general 1-letter prefixes.
    """
    brain_regions = []
    
    # Ensure these keys exist in your encoding dictionary
    # If your model doesn't support 'central', map 'C'/'CP' to 'parietal' or 'frontal'
    # BRAIN_REGION_ENCODING = {"frontal": 0, "temporal": 1, "parietal": 2, "occipetal": 3, "central": 4} 

    for channel_name in channel_names:
        name = channel_name.upper()
        
        # 1. Check specific multi-letter prefixes first to avoid ambiguity
        if name.startswith("FP") or name.startswith("AF"):
            brain_regions.append(BRAIN_REGION_ENCODING["frontal"])
        
        elif name.startswith("FC"):
            # Fronto-Central: Map to Frontal (or Central depending on your preference)
            brain_regions.append(BRAIN_REGION_ENCODING["frontal"])
            
        elif name.startswith("FT"):
            # Fronto-Temporal: Map to Temporal
            brain_regions.append(BRAIN_REGION_ENCODING["temporal"])
            
        elif name.startswith("TP"):
            # Temporo-Parietal: Map to Temporal (or Parietal)
            brain_regions.append(BRAIN_REGION_ENCODING["temporal"])
            
        elif name.startswith("CP"):
            # Centro-Parietal: Map to Central (or Parietal)
            brain_regions.append(BRAIN_REGION_ENCODING["central"])
            
        elif name.startswith("PO"):
            # Parieto-Occipital: Map to Occipital (or Parietal)
            brain_regions.append(BRAIN_REGION_ENCODING["occipetal"])
            
        elif name.startswith("IZ"):
            # Inion: Map to Occipital
            brain_regions.append(BRAIN_REGION_ENCODING["occipetal"])

        # 2. Check general single-letter prefixes
        elif name.startswith("F"):
            # Handles F1, F2, Fz, etc.
            brain_regions.append(BRAIN_REGION_ENCODING["frontal"])
            
        elif name.startswith("T"):
            # Handles T7, T8, etc.
            brain_regions.append(BRAIN_REGION_ENCODING["temporal"])
            
        elif name.startswith("C"):
            # Handles C1, C2, Cz, etc.
            brain_regions.append(BRAIN_REGION_ENCODING["central"])
            
        elif name.startswith("P"):
            # Handles P1, P2, Pz, etc.
            brain_regions.append(BRAIN_REGION_ENCODING["parietal"])
            
        elif name.startswith("O"):
            # Handles O1, O2, Oz
            brain_regions.append(BRAIN_REGION_ENCODING["occipetal"])
            
        else:
            # Fallback to ensure length matches (prevents dropping)
            print(f"Warning: Channel {name} unmapped. Defaulting to Central.")
            brain_regions.append(BRAIN_REGION_ENCODING["central"])

    return brain_regions
 
class CSBrain(nn.Module):
    def __init__(self, in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                 nhead=8, TemEmbed_kernel_sizes=[(1,), (3,), (5,)], brain_regions=[], sorted_indices=[]):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)

        self.TemEmbed_kernel_sizes = TemEmbed_kernel_sizes
        kernel_sizes = self.TemEmbed_kernel_sizes
        self.TemEmbedEEGLayer = TemEmbedEEGLayer(dim_in=in_dim, dim_out=out_dim, kernel_sizes=kernel_sizes, stride=1)

        self.brain_regions = brain_regions
        self.area_config = generate_area_config(sorted(brain_regions))
        self.BrainEmbedEEGLayer = BrainEmbedEEGLayer(dim_in=in_dim, dim_out=out_dim)
        self.sorted_indices = sorted_indices

        encoder_layer = CSBrain_TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, area_config=self.area_config, sorted_indices=self.sorted_indices, batch_first=True,
            activation=F.gelu
        )
        self.encoder = CSBrain_TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)

        self.proj_out = nn.Sequential(
            nn.Linear(d_model, out_dim),
        )
        self.apply(_weights_init)

        self.features_by_layer = []
        self.input_features = []

    def forward(self, x, mask=None):
        x = x[:, self.sorted_indices, :, :]

        patch_emb = self.patch_embedding(x, mask)

        for layer_idx in range(self.encoder.num_layers):
            patch_emb = self.TemEmbedEEGLayer(patch_emb) + patch_emb
            patch_emb = self.BrainEmbedEEGLayer(patch_emb, self.area_config) + patch_emb

            patch_emb = self.encoder.layers[layer_idx](patch_emb, self.area_config)

        out = self.proj_out(patch_emb)

        return out


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3),
                      groups=d_model),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(d_model // 2 + 1, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        mask_x = mask_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, mask_x.shape[1] // 2 + 1)
        spectral_emb = self.spectral_proj(spectral)
        patch_emb = patch_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def generate_area_config(brain_regions):
    region_to_channels = defaultdict(list)
    for channel_idx, region in enumerate(brain_regions):
        region_to_channels[region].append(channel_idx)

    area_config = {}
    for region, channels in region_to_channels.items():
        area_config[f'region_{region}'] = {
            'channels': len(channels),
            'slice': slice(channels[0], channels[-1] + 1)
        }
    return area_config

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

class csbrain_classification(BaseModel):
    def __init__(
            self,
            in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
            nhead=8, TemEmbed_kernel_sizes=[(1,), (3,), (5,)], brain_regions=[], sorted_indices=[],
            n_classes: int=None, classifier_type: str="all_patch_reps_onelayer", dropout: float=0.1, 
            n_channels: int=64, n_seconds_input: int=4, 
            finetune_type: str="full_finetune",
            **kwargs
    ):
        super().__init__()
        self.in_dim = in_dim

        base_backbone = CSBrain(in_dim=in_dim, out_dim=out_dim, dim_feedforward=dim_feedforward, seq_len=seq_len,
                                n_layer=n_layer, nhead=nhead, TemEmbed_kernel_sizes=TemEmbed_kernel_sizes, brain_regions=brain_regions,
                                sorted_indices=sorted_indices)
        
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_seconds_input = n_seconds_input
        self.finetune_type = finetune_type

        if kwargs["pretrained_checkpoint"] is not None:
            pretrained_checkpoint = kwargs['pretrained_checkpoint']
            if not os.path.exists(pretrained_checkpoint):
                pretrained_checkpoint = os.path.join(os.getenv("BIODL_PRETRAINED_CHECKPOINT_DIR", "./pretrain_checkpoints"), "csbrain", pretrained_checkpoint)

            # load checkpoint for CSBrain backbone
            map_location = torch.device("cuda:0")
            state_dict = torch.load(pretrained_checkpoint, map_location=map_location)

            new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

            model_state_dict = base_backbone.state_dict()

            # find the matching set of weights
            matching_dict = {k: v for k, v in new_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
            model_state_dict.update(matching_dict)
            base_backbone.load_state_dict(model_state_dict)

        base_backbone.proj_out = nn.Identity()

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
        
        if finetune_type == "linear_probe":
            print(f"Freezing parameters of CSBrain encoder")
            for p in self.backbone.parameters():
                p.requires_grad = False
                
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
                nn.Linear(int(self.n_channels * n_seconds_input * 200), n_classes),
            )
        elif classifier_type == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(self.n_channels * n_seconds_input * 200, 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(200, n_classes),
            )
        elif classifier_type == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(self.n_channels * n_seconds_input * 200, n_seconds_input * 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(n_seconds_input * 200, 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(200, n_classes),
            )
    
    def train(self, mode=True):
        super().train(mode)

        if self.finetune_type == "linear_probe":
            self.backbone.eval()

    def forward(self, x, labels=None, return_embeddings: bool=False, **kwargs):
        loss = None 
        # reshape x: (batch_size, n_channels, T) -> (batch_size, n_channels, n_patches, patch_size)
        B, n_channels, T = x.shape

        x = x.view(B, n_channels, -1, self.in_dim)
        feats = self.backbone(x)

        logits = self.classifier(feats)

        if labels is not None:
            loss = self.loss_function(logits, labels)
        
        return {
            "loss": loss,
            "logits": logits,
            "embeddings": feats if return_embeddings else None
        }
    
    def get_optimizer_params(self, weight_decay: float):
        """
        Separates parameters into groups with and without weight decay.
        
        Excludes from weight decay:
        1. The 'classifier' head (as requested).
        2. Biases (standard practice).
        3. Normalization parameters (GroupNorm, BatchNorm).
        4. LoRA parameters (optional, but often recommended for stability).
        """
        decay = []
        no_decay = []

        # Iterating named_parameters returns only the params registered in the model
        for name, param in self.named_parameters():
            # 1. Skip frozen parameters (handles linear_probe and frozen parts of LoRA)
            if not param.requires_grad:
                continue
            
            # 2. Identify parameters that should NOT have weight decay
            # - "classifier": Your specific requirement
            # - "bias": Biases should not be regularized
            # - "norm" / "GroupNorm": Normalization scales/shifts
            # - "lora": (Optional) Often safer to not decay LoRA adapters in low-data regimes
            # - len(param.shape) == 1: Catch-all for 1D params (biases, layer norms)
            if (
                "classifier" in name 
                or "bias" in name 
                or "norm" in name 
                or "lora" in name 
                or len(param.shape) == 1
            ):
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}
        ]


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CSBrain(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                    nhead=8).to(device)
    model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth',
                                     map_location=device))
    a = torch.randn((8, 16, 10, 200)).cuda()
    b = model(a)
    print(a.shape, b.shape)