import os, sys, json
import numpy as np 
import torch 


# Common EEG aliases (New Terminology -> Old Terminology)
# We map common older names to the standard 10-20 names used in BIOT
ALIASES = {
    'T7': 'T3', 'T8': 'T4',
    'P7': 'T5', 'P8': 'T6',
    'Fp1': 'FP1', 'Fp2': 'FP2', # normalization just in case
    'FP1-F3': 'Fpz-Cz', 
    'F3-C3':  'Fpz-Cz', 
    'P3-O1':  'Pz-Oz',  
    
    'FP2-F4': 'Fpz-Cz',
    'F4-C4':  'Fpz-Cz',
    'P4-O2':  'Pz-Oz',
}


# Helper to find index of a channel, checking exact name and aliases
def find_channel_index(target_name, channel_list, channel_map=None):
    if target_name in channel_list:
        return channel_list.index(target_name)
    
    # Check reverse alias (if we need T7 but have T3)
    if target_name in ALIASES and ALIASES[target_name] in channel_list:
        return channel_list.index(ALIASES[target_name])
        
    # Check if the input list uses the alias instead (if we need T3 but have T7)
    # Invert the alias map for this check
    rev_aliases = {v: k for k, v in ALIASES.items()}
    if target_name in rev_aliases and rev_aliases[target_name] in channel_list:
            return channel_list.index(rev_aliases[target_name])
            
    return None

def process_labram(data, input_channels):
    mapping_channels = {
        "Fpz-Cz": "F4",
        "Pz-Oz": "P4",
        "P08": "PO8"
    }

    output_channels = []
    for i in input_channels:
        if i in mapping_channels.keys():
            output_channels.append(mapping_channels[i])
        else:
            output_channels.append(i)

    # divide by 100 to convert uV -> 100uv such that [-100uv, 100uv] maps to [-1, 1]
    # In order for this to work, we need to ensure that the data is read in uV in the data processing scripts
    return data/100, output_channels

def process_eegfm(data, input_channels, **kwargs):
    # import mne 
    # montage_channels = sorted([ch.lower() for ch in mne.channels.make_standard_montage('standard_1020').ch_names])
    montage_channels = sorted(json.load(open(os.path.join(os.getenv("BIODL_RAW_DATA_ROOT"), "tueg", "v2.0.1", "DOCS", "montage_channels_10_20.json"), "r")))
    
    output_channels, output_channels_index = [], []
    channel_idx = []
    norm_type = kwargs.get("norm_type")

    for i in input_channels:
        chan = f"EEG {i.upper()}-REF"
        if chan in montage_channels:
            output_channels.append(i.lower())
            output_channels_index.append(input_channels.index(i))
            channel_idx.append(montage_channels.index(chan))

    data = data[output_channels_index]
    if norm_type == "uv_norm":
        data = data / 100.0

    elif norm_type == "z_norm_file":
        mean = np.mean(data)
        std = np.std(data)
        data = (data - mean) / (std + 1e-8)

    elif norm_type == "z_norm_per_channel":
        # Keep dims to broadcast back to (channels, time)
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        data = (data - mean) / (std + 1e-8)

    elif norm_type == "percentile_norm_file":
        p5 = np.percentile(data, 5)
        p95 = np.percentile(data, 95)
        # Robust scaling based on 5th and 95th percentiles
        data = (data - p5) / ((p95 - p5) + 1e-8)

    elif norm_type == "percentile_norm_per_channel":
        p5 = np.percentile(data, 5, axis=1, keepdims=True)
        p95 = np.percentile(data, 95, axis=1, keepdims=True)
        data = (data - p5) / ((p95 - p5) + 1e-8)
    # import pdb; pdb.set_trace()
    return {"data": data, "channel_idx": channel_idx}, output_channels

def process_cbramod(data, input_channels):
    # Standardize input channels to uppercase to avoid case-sensitivity issues
    mapping_channels = {
        "Fpz-Cz": "Fz",
        "Pz-Oz": "Pz",
        "Fc3": "F7",
        "Fc1": "F3",
        "Fc2": "F4",
        "Fc4": "F8",
        "P1": "P3",
        "P2": "P4",
        "POz": "O1",
        "P08": "PO8"
    }

    cached_channels = []
    for i in input_channels:
        if i in mapping_channels.keys():
            cached_channels.append(mapping_channels[i])
        else:
            cached_channels.append(i)

    input_channels = [ch.upper().strip() for ch in cached_channels]
    new_data = data 
    out_channels = input_channels

    return new_data/100, out_channels

def process_ssl_data(data, input_channels, ssl, **kwargs):
    if "cbramod" in ssl:
        return process_cbramod(data, input_channels)
    elif "labram" in ssl:
        return process_labram(data, input_channels)
    elif "csbrain" in ssl:
        return process_cbramod(data, input_channels)
    elif "eegfm" in ssl:
        return process_eegfm(data, input_channels, **kwargs)