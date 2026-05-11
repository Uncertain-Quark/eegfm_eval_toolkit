from eegfm_eval_toolkit.models.eegnet import EEGNet
from eegfm_eval_toolkit.models.eegnet_ae import EEGNetAutoEncoder
from eegfm_eval_toolkit.models.eegnex import EEGNeX
from eegfm_eval_toolkit.models.mae import EEGMAE
from eegfm_eval_toolkit.models.sparcnet import SPaRCNet
from eegfm_eval_toolkit.models.eegtransformer import ScalableEEGTransformer
from eegfm_eval_toolkit.models.biot import BIOTSupervisedPretrain
from eegfm_eval_toolkit.models.cbramod import CBraMod, CBraMod_Classification
from eegfm_eval_toolkit.models.labram import get_input_chans, NeuralTransformer
from eegfm_eval_toolkit.models.labram import load_state_dict as load_state_dict_labram
from eegfm_eval_toolkit.models.csbrain import get_brain_regions, get_sorted_indices, csbrain_classification
from eegfm_eval_toolkit.models.eegfm import EEGFM_MAE, EEGFM_Contrastive, EEGFM_BYOL

import torch 
import torch.nn as nn
from functools import partial

def get_models(model_type, model_params=None, channel_names=None, **kwargs):
    input_chans = None
    if model_type == "eegnet":
        model = EEGNet(**model_params)
    elif model_type == "eegnet_ae":
        model = EEGNetAutoEncoder(**model_params)
        if "pretrained_checkpoint" in model_params:
            print(f"Loading from checkpoint: {model_params['pretrained_checkpoint']}")
            # load the pretrained checkpoint
            state_dict = torch.load(model_params["pretrained_checkpoint"])
            model_state_dict = state_dict["model_state_dict"]

            model_dict = model.state_dict()
            model_state_dict = {
                k:v for k,v in model_state_dict.items()
                if k in model_dict and v.size() == model_dict[k].size()
            }

            model.load_state_dict(model_state_dict, strict=False)
            
    elif model_type == "eegnet_large":
        model = EEGNet(**model_params, F1=16, D=4)
    elif model_type == "eegnet_huge":
        model = EEGNet(**model_params, F1=32, D=8)
    elif "eegnex" in model_type:
        parts = model_type.split("_")
        f1, f2 = int(parts[1]), int(parts[2])
        print(f"Initializing EEGNeX model with f1={f1} f2={f2}")
        model = EEGNeX(**model_params, f1=f1, f2=f2)
    elif "eegtransformer" in model_type:
        model = ScalableEEGTransformer(**model_params)
        n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
        print(f"Training parameters in EEG Transformer: {n_params/1e6:.3f}M")
    elif model_type == "eegmae":
        model = EEGMAE(**model_params)
    elif model_type == "sparcnet":
        model = SPaRCNet(**model_params)
    elif model_type == "cbramod":
        model = CBraMod(**model_params)
    elif model_type == "cbramod_classifier":
        model = CBraMod_Classification(**model_params)
    elif model_type == "labram_base":
        model = NeuralTransformer(
            patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.1, **model_params
        )
        input_chans = get_input_chans(channel_names)
    elif model_type == "csbrain":

        # get the brain regions
        brain_regions = get_brain_regions(channel_names)
        sorted_indices = get_sorted_indices(channel_names, brain_regions)

        model = csbrain_classification(
            **model_params,
            brain_regions=brain_regions,
            sorted_indices=sorted_indices
        )
    else:
        raise ValueError(f"Unknown model name: {model_type} passed")
    
    return {
        "model": model,
        "input_chans": input_chans
    }
