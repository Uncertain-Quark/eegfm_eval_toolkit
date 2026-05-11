# Implementation taken from: https://github.com/935963004/LaBraM/blob/main/modeling_finetune.py
# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------
import os 

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
# from timm.models.registry import register_model
from einops import rearrange

from biodl.models.base import BaseModel
from peft import get_peft_model, LoraConfig, TaskType
from einops.layers.torch import Rearrange


standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def get_input_chans(ch_names):
    input_chans = [0] # for cls token
    for ch_name in ch_names:
        input_chans.append(standard_1020.index(ch_name.upper()) + 1)
    return input_chans


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_norm=None, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if qk_norm is not None:
            self.q_norm = qk_norm(head_dim)
            self.k_norm = qk_norm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) (B, H, N, C)
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn
            
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_qkv:
            return x, qkv

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        if return_attention:
            return self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True)
        if return_qkv:
            y, qkv = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_qkv=return_qkv)
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x, qkv

        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ EEG to Patch Embedding
    """
    def __init__(self, EEG_size=2000, patch_size=200, in_chans=1, embed_dim=200):
        super().__init__()
        # EEG_size = to_2tuple(EEG_size)
        # patch_size = to_2tuple(patch_size)
        num_patches = 62 * (EEG_size // patch_size)
        self.patch_shape = (1, EEG_size // patch_size)
        self.EEG_size = EEG_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TemporalConv(nn.Module):
    """ EEG to Patch Embedding
    """
    def __init__(self, in_chans=1, out_chans=8):
        '''
        in_chans: in_chans of nn.Conv2d()
        out_chans: out_chans of nn.Conv2d(), determing the output dimension
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        x = rearrange(x, 'B N A T -> B (N A) T')
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        return x

class PeftCompatWrapper(nn.Module):
    """
    Wraps a custom model to make it compatible with PEFT's expected input signature.
    PEFT expects 'input_ids', but CBraMod expects 'x'.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def __getattr__(self, name):
        """Forward missing attributes (like .blocks) to the wrapped model."""
        try:
            # First, try to get the attribute from the wrapper itself (or nn.Module)
            return super().__getattr__(name)
        except AttributeError:
            # If not found, look in the internal wrapped model
            return getattr(self.model, name)
        
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
        if "input_chans" in kwargs:
            valid_args["input_chans"] = kwargs["input_chans"]
        
        if "return_all_tokens" in kwargs:
            valid_args["return_all_tokens"] = kwargs["return_all_tokens"] 
        if "return_patch_tokens" in kwargs:
            valid_args["return_patch_tokens"] = kwargs["return_patch_tokens"]
            
        return self.model(inputs, **valid_args)

class ViTEncoder(nn.Module):
    """
    Helper class to hold the backbone components (Embeddings + Transformer Blocks).
    This makes it easy to wrap with LoRA or freeze independently.
    """
    def __init__(self, EEG_size, patch_size, in_chans, embed_dim, depth, num_heads, 
                 mlp_ratio, qkv_bias, qk_norm, qk_scale, drop_rate, attn_drop_rate, 
                 drop_path_rate, norm_layer, init_values, use_abs_pos_emb, 
                 use_rel_pos_bias, use_shared_rel_pos_bias, use_mean_pooling, 
                 out_chans):
        super().__init__()
        self.patch_size = patch_size
        self.time_window = EEG_size // patch_size
        self.embed_dim = embed_dim

        # Patch Embedding
        # Assuming TemporalConv and PatchEmbed are defined elsewhere in your code
        self.patch_embed = TemporalConv(out_chans=out_chans) if in_chans == 1 else PatchEmbed(EEG_size=EEG_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # Tokens & Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim), requires_grad=True)
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                qk_norm=qk_norm, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values, window_size=None)
            for i in range(depth)])
        
        # Norms
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None

        # Init weights
        if self.pos_embed is not None: trunc_normal_(self.pos_embed, std=.02)
        if self.time_embed is not None: trunc_normal_(self.time_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_patch_tokens=False, return_all_tokens=False, input_chans=None, **kwargs):
        # import pdb;pdb.set_trace()
        batch_size, n, a, t = x.shape
        input_time_window = a if t == self.patch_size else t
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
        if self.pos_embed is not None:
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, input_time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((pos_embed_used[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        
        # Interpolation code for sequences 
        if self.time_embed is not None:
            t_embed = self.time_embed 
            max_len = self.time_embed.shape[1]

            if input_time_window > max_len:
                # Permute to (1, embed_dim, 16) for interpolation
                t_embed = t_embed.permute(0, 2, 1)
                
                # Resize from 16 -> input_time_window (e.g. 20)
                t_embed = torch.nn.functional.interpolate(
                    t_embed, 
                    size=input_time_window, 
                    mode='linear',  # or 'bicubic'
                    align_corners=False
                )
                
                # Permute back to (1, 20, embed_dim)
                t_embed = t_embed.permute(0, 2, 1)
            else:
                # If input is shorter (e.g. 10s), just slice it as before
                t_embed = t_embed[:, :input_time_window, :]

            nc = n if t == self.patch_size else a
            time_embed = t_embed.unsqueeze(1).expand(batch_size, nc, -1, -1).flatten(1, 2)
            
            x[:, 1:, :] += time_embed

        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=None)
        
        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens: return self.fc_norm(x)
            t = x[:, 1:, :]
            if return_patch_tokens: return self.fc_norm(t)
            else: return self.fc_norm(t.mean(1))
        else:
            if return_all_tokens: return x
            elif return_patch_tokens: return x[:, 1:]
            else: return x[:, 0]

class AttentionPoolingHead(nn.Module):
    def __init__(self, embed_dim, n_classes, hidden_dim=None):
        super().__init__()
        # 1. Attention mechanism
        # Takes features (B, N, D) and produces weights (B, N, 1)
        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim or embed_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim or embed_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # 2. Final Linear Probe
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        # x shape: (Batch, N_tokens, Embed_dim)
        
        # Calculate weights: "Which tokens are interesting?"
        weights = self.attention_net(x) 
        
        # Weighted average: (Batch, N, D) * (Batch, N, 1) -> Sum -> (Batch, D)
        context_vector = torch.sum(x * weights, dim=1)
        
        return self.classifier(context_vector)

class AdaptiveProbeHead(nn.Module):
    def __init__(self, embed_dim, n_classes, target_seq_len=64):
        super().__init__()
        # Reduces sequence length from N (e.g. 1600) to target_seq_len (e.g. 16)
        self.pool = nn.AdaptiveAvgPool1d(target_seq_len)
        self.norm = nn.LayerNorm(embed_dim * target_seq_len)
        self.classifier = nn.Linear(embed_dim * target_seq_len, n_classes)

    def forward(self, x):
        # x shape: (Batch, N_tokens, Embed_dim)
        # Permute for pooling: (Batch, Embed_dim, N_tokens)
        x = x.transpose(1, 2)
        
        # Pool: (Batch, Embed_dim, 16)
        x = self.pool(x)
        
        # Flatten: (Batch, Embed_dim * 16)
        x = x.flatten(1)
        
        x = self.norm(x)
        return self.classifier(x)
    
class NeuralTransformer(BaseModel):
    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, n_classes=1000, embed_dim=200, depth=12,
                 num_heads=10, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001, sampling_rate=None, n_seconds_input=None,
                 finetune_type: str="full_finetune", classifier_type: str="linear_probe_all_tokens", **kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        EEG_size = int(n_seconds_input*sampling_rate)
        self.finetune_type = finetune_type
        
        # 1. Initialize the Backbone (Encoder)
        self.backbone = ViTEncoder(
            EEG_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, 
            qkv_bias, qk_norm, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, 
            norm_layer, init_values, use_abs_pos_emb, use_rel_pos_bias, 
            use_shared_rel_pos_bias, use_mean_pooling, out_chans
        )

        self.return_all_tokens, self.return_patch_tokens = False, False
        # 2. Initialize Head
        if classifier_type == "linear_probe_all_tokens":
            self.head = nn.Sequential(
                Rearrange('b s d -> b (s d)'),
                nn.Linear(int(kwargs['n_channels']*embed_dim*n_seconds_input) + embed_dim, n_classes),
            )
            self.return_all_tokens=True
        elif classifier_type == "linear_probe_attention":
            self.head = AttentionPoolingHead(embed_dim, n_classes)
            self.return_patch_tokens = True
        elif classifier_type == "linear_probe_adaptive":
            self.head = AdaptiveProbeHead(embed_dim, n_classes)
            self.return_patch_tokens = True
        else:
            self.head = nn.Linear(embed_dim, n_classes) if n_classes > 0 else nn.Identity()
            if isinstance(self.head, nn.Linear):
                trunc_normal_(self.head.weight, std=.02)
                self.head.weight.data.mul_(init_scale)
                if self.head.bias is not None:
                    self.head.bias.data.mul_(init_scale)
            

        # 3. Load Checkpoint (With remapping to 'backbone.')
        if "pretrained_checkpoint" in kwargs:
            self._load_pretrained(kwargs["pretrained_checkpoint"])

        # 4. Handle LoRA Injection
        if finetune_type == "lora":
            print("Injecting LoRA adapters into the backbone...")
            # Default target modules for ViT usually include qkv/proj/fc1/fc2
            # Adjust these names based on your Block implementation (e.g. "qkv", "fc1", "fc2", "proj")
            self.backbone = PeftCompatWrapper(self.backbone)
            target_modules = kwargs.get("target_modules", ["fc1", "fc2"]) 
            
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, # Generic task type since we wrap just the backbone
                inference_mode=False, 
                r=kwargs.get("lora_r", 4), 
                lora_alpha=kwargs.get("lora_alpha", 8), 
                lora_dropout=kwargs.get("lora_dropout", 0.1),
                target_modules=target_modules
            )
            # Wrap ONLY the backbone
            self.backbone = get_peft_model(self.backbone, peft_config)
            self.backbone.print_trainable_parameters()

        # 5. Handle Linear Probe (Freezing)
        elif "linear_probe" in finetune_type:
            print("Finetune type is 'linear_probe'. Freezing backbone.")
            for param in self.backbone.parameters():
                param.requires_grad = False


    def _load_pretrained(self, ckpt_path):
        if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(os.getenv("BIODL_PRETRAINED_CHECKPOINT_DIR", "./pretrain_checkpoints"), "labram", ckpt_path)
        print(f"Loading pretrained checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        checkpoint_model = checkpoint.get("model", checkpoint.get("module", checkpoint))
        
        new_state_dict = {}
        for k, v in checkpoint_model.items():
            # Standard cleaning
            k = k.replace("student.", "")
            if k.startswith("norm."): k = k.replace("norm.", "fc_norm.")
            
            # Remap head
            if k.startswith("lm_head."): k = k.replace("lm_head.", "head.")
            
            # Remap backbone keys: if it's not the head, it belongs to backbone
            if not k.startswith("head."):
                k = f"backbone.{k}"
            
            new_state_dict[k] = v

        # Handle Head Shape Mismatch
        current_state_dict = self.state_dict()
        for k in ['head.weight', 'head.bias']:
            if (k not in current_state_dict) or (k in new_state_dict and new_state_dict[k].shape != current_state_dict[k].shape):
                print(f"Removing key {k} due to shape mismatch")
                del new_state_dict[k]
        
        msg = self.load_state_dict(new_state_dict, strict=False)
        print(f"Weights loaded. Missing: {msg.missing_keys}")

    def train(self, mode=True):
        super().train(mode)

        if  "linear_probe" in self.finetune_type:
            self.backbone.eval()

    def forward(self, x, labels=None, return_embeddings=False, **kwargs):
        # Reshape logic
        B, n_c, T = x.shape
        x = x.view(B, n_c, -1, self.patch_size)

        # Forward through backbone
        # Note: input_chans and return_all_tokens logic handled in ViTEncoder
        features = self.backbone(x, return_all_tokens=self.return_all_tokens, return_patch_tokens=self.return_patch_tokens, **kwargs)
        logits = self.head(features)
        
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "embeddings": features if return_embeddings else None
        }

    # Proxy methods to maintain API compatibility
    def get_num_layers(self):
        return len(self.backbone.blocks)
    
    def get_optimizer_params(self, weight_decay: float, lr: float, layer_decay: float = 0.65):
        """
        Combines Layer-wise Learning Rate Decay (LLRD) with Weight Decay exclusion.
        
        Args:
            weight_decay: The base weight decay (e.g., 0.05).
            lr: The peak learning rate (e.g., 5e-4).
            layer_decay: The decay factor (e.g., 0.65).
        """
        num_layers = self.get_num_layers()
        # parameter_groups will store: {(lr_value, wd_value): [list_of_params]}
        parameter_groups = {}

        skip_keywords = self.no_weight_decay()

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # 1. Determine Weight Decay for this parameter
            # Keep your existing logic: No decay for head, biases, norms, or embeddings
            if (
                "head" in name 
                or "bias" in name 
                or "norm" in name 
                or "lora" in name 
                or len(param.shape) == 1 
                or name in skip_keywords
            ):
                this_wd = 0.0
            else:
                this_wd = weight_decay

            # 2. Determine Layer ID for Learning Rate Decay
            # Layer IDs: Stem=0, Block_i=i+1, Head=num_layers+1
            if name.startswith("head"):
                layer_id = num_layers + 1
            elif name.startswith("backbone.blocks."):
                # Extract block index from "backbone.blocks.0.norm1.weight" -> 0
                layer_id = int(name.split('.')[2]) + 1
            else:
                # Stem / Patch Embed / Pos Embed
                layer_id = 0

            # 3. Calculate LR for this specific layer
            # Formula: peak_lr * (layer_decay ** (distance_from_head))
            this_lr = lr * (layer_decay ** (num_layers + 1 - layer_id))

            # 4. Group parameters by the unique combination of (LR, WD)
            group_key = (this_lr, this_wd)
            if group_key not in parameter_groups:
                parameter_groups[group_key] = []
            parameter_groups[group_key].append(param)

        # Convert dictionary to the list format expected by torch.optim
        return [
            {"params": params, "lr": l, "weight_decay": w}
            for (l, w), params in parameter_groups.items()
        ]

    def no_weight_decay(self):
        return {'backbone.pos_embed', 'backbone.cls_token', 'backbone.time_embed'}



