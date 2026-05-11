import os, sys, json
import mne 
import numpy as np
import pandas as pd 
import lmdb 
import pickle 
import glob
import tqdm
from sklearn.model_selection import StratifiedGroupKFold

from biodl.utils.make_ssl_data import process_ssl_data

# ==========================================
# 1. Classification Paradigms (from Paper)
# ==========================================
# Original: 1:SPSW, 2:GPED, 3:PLED, 4:EYEM, 5:ARTF, 6:BCKG

# 4-WAY: SPSW(0), GPED(1), PLED(2), BACKG(3) (combines EYEM, ARTF, BCKG)
LABEL_MAP_4WAY = {1:0, 2:1, 3:2, 4:3, 5:3, 6:3} 

# 2-WAY: TARG(0) (combines SPSW, GPED, PLED), BCKG(1) (combines EYEM, ARTF, BCKG)
LABEL_MAP_2WAY = {1:0, 2:0, 3:0, 4:1, 5:1, 6:1} 

# Priority for Stratification (Rare events have lower numbers/higher priority)
CLASS_PRIORITY = {
    2: 0,  # GPED
    3: 1,  # PLED
    1: 2,  # SPSW
    4: 3,  # EYEM
    5: 4,  # ARTF
    6: 5   # BCKG
}

# ==========================================
# 2. Helper Functions
# ==========================================

def get_stratification_label(unique_labels_in_file):
    """Returns the single label ID that is 'rarest' in the file for splitting purposes."""
    if not unique_labels_in_file:
        return 6 # Default to Background
    # Sort labels by priority 
    best_label = sorted(unique_labels_in_file, key=lambda x: CLASS_PRIORITY.get(x, 100))[0]
    return best_label

def convert_to_tcp_montage(raw):
    # (Same as your original function)
    TCP_MONTAGE = ["FP1-F7","F7-T3","T3-T5","T5-O1","FP2-F8","F8-T4","T4-T6","T6-O2","A1-T3","T3-C3","C3-CZ","CZ-C4","C4-T4","T4-A2","FP1-F3","F3-C3","C3-P3","P3-O1","FP2-F4","F4-C4","C4-P4","P4-O2"]
    ch_names = raw.info["ch_names"]
    MODE = raw.info["ch_names"][0].split("-")[-1]
    eeg_data = raw.get_data()
    tcp_eeg_data = np.zeros((len(TCP_MONTAGE), eeg_data.shape[-1]))
    for i, ch in enumerate(TCP_MONTAGE):
        ch_1, ch_2 = ch.split("-")
        try:
            tcp_eeg_data[i] = eeg_data[ch_names.index(f"EEG {ch_1}-{MODE}")] - eeg_data[ch_names.index(f"EEG {ch_2}-{MODE}")]
        except ValueError:
            pass # Handle missing channels if necessary
    tcp_mne_info = mne.create_info(ch_names=TCP_MONTAGE, sfreq=raw.info["sfreq"], ch_types=["eeg"]*len(TCP_MONTAGE))
    return mne.io.RawArray(tcp_eeg_data, tcp_mne_info)

def convert_to_ssl_montage(raw):
    # (Same as your original function)
    SSL_MONTAGE = ["FP1","F7","T3","T5","O1","FP2","F8","T4","T6","O2","C3","CZ","C4","F3","P3","F4","P4"]
    ch_names = raw.info["ch_names"]
    MODE = raw.info["ch_names"][0].split("-")[-1]
    eeg_data = raw.get_data()
    ssl_eeg_data = np.zeros((len(SSL_MONTAGE), eeg_data.shape[-1]))
    for i, ch in enumerate(SSL_MONTAGE):
        ssl_eeg_data[i] = eeg_data[ch_names.index(f"EEG {ch}-{MODE}")]
    ssl_mne_info = mne.create_info(ch_names=SSL_MONTAGE, sfreq=raw.info["sfreq"], ch_types=["eeg"]*len(SSL_MONTAGE))
    return mne.io.RawArray(ssl_eeg_data, ssl_mne_info)

def get_paths():
    root = os.getenv("BIODL_RAW_DATA_ROOT", "./data")
    data_root = os.path.join(root, "tuev") 
    pre_root = os.getenv("BIODL_PREPROCESSED_DATA_ROOT", "./data_preprocessed")
    split_path = os.path.join(os.getenv("BIODL_SPLIT_PATH", "./splits"), "tuev_1s")
    os.makedirs(split_path, exist_ok=True)
    return data_root, pre_root, split_path

# ==========================================
# 3. Main Processing Logic
# ==========================================

def process_single_file(fpath, config):
    duration_window = config.get("duration", 1.0)
    raw = mne.io.read_raw_edf(fpath, preload=True)
    raw.set_eeg_reference(ref_channels='average')
    
    # Montage Conversion
    if "ssl" not in config.keys():
        raw_tcp = convert_to_tcp_montage(raw)
    else:
        raw_tcp = convert_to_ssl_montage(raw)
    
    eeg_channels = raw_tcp.info["ch_names"]
    
    # Resample
    if raw.info['sfreq'] != config["fs"]:
        raw_tcp.resample(config['fs'], verbose=False, method="polyphase", n_jobs=8)

    # Preprocessing: Notch & Bandpass
    raw_tcp.notch_filter(np.arange(60, raw_tcp.info['sfreq']/2, 60), verbose=False)
    raw_tcp.filter(l_freq=0.3, h_freq=75, verbose=False)
    
    if len(raw_tcp.info['bads']) > 0:
        raw_tcp.interpolate_bads()
    
    # Extract continuous data array
    full_data = raw_tcp.get_data(units="uV").astype(np.float32)

    if config.get('ssl', None) is not None:
        # print(f"Entered SSL loop: {cfg.get('ssl')}")
        full_data, eeg_channels = process_ssl_data(full_data, eeg_channels, ssl=config['ssl'])
        # if "channels" not in storage[cfg['name']]:
        #     storage[cfg['name']]['channels'] = out_channels

    # Normalize (Z-score)
    if config["norm"]:
        mu = np.mean(full_data, axis=1, keepdims=True)
        std = np.std(full_data, axis=1, keepdims=True)
        std[std == 0] = 1.0 
        full_data = (full_data - mu) / std
    
    elif config.get("robust"):
        # Robust Scaling (Quantile-based)
        # We use the 25th (Q1) and 75th (Q3) percentiles
        q1 = np.percentile(full_data, 25, axis=1, keepdims=True)
        q3 = np.percentile(full_data, 75, axis=1, keepdims=True)
        median = np.median(full_data, axis=1, keepdims=True)
        iqr = q3 - q1
        
        # Guard against flat channels/zero division
        iqr[iqr == 0] = 1.0
        
        full_data = (full_data - median) / iqr
        
        # Optional: Clipping
        # After robust scaling, extreme artifacts might still be +/- 100.
        # Clipping to [-20, 20] often helps neural networks converge faster.
        full_data = np.clip(full_data, -20, 20)
    
    # --- EVENT BASED PARSING ---
    key = os.path.basename(fpath).replace(".edf", "")
    rec_path = fpath.replace('.edf', '.rec')
    
    # Read Annotations
    annotations = pd.read_csv(rec_path, delimiter=',', header=None, comment='#')
    # Handle optional confidence column
    annotations.columns = ['channel', 'start', 'end', 'label'] if annotations.shape[1] == 4 else ['channel', 'start', 'end', 'label', 'conf']
    
    samples = list()
    classes_present = set()

    # Iterate through specific events rather than sliding 5s windows
    for _, row in annotations.iterrows():
        label_6way = int(row['label'])
        classes_present.add(label_6way)
        
        # Calculate indices
        start_idx = int(row['start'] * config['fs'])
        end_idx = int(row['end'] * config['fs'])
        
        # Duration check: Ensure minimum 1 second (fs samples) for stability
        # If segment is < 1s, center it and expand
        if (end_idx - start_idx) < int(config['fs']*duration_window):
            mid = (start_idx + end_idx) // 2
            start_idx = int(mid - (int(config['fs']*duration_window) // 2))
            end_idx = int(mid + (int(config['fs']*duration_window) // 2))

        # Boundary checks
        if start_idx < 0: start_idx = 0
        if end_idx > full_data.shape[1]: end_idx = full_data.shape[1]
        
        # Skip if still invalid
        if end_idx - start_idx <= 0: continue

        # Extract the specific event segment
        segment_data = full_data[:, start_idx:end_idx]
        # import pdb; pdb.set_trace()

        samples.append({
            "data": segment_data, # The actual EEG numbers for this event
            "start": row['start'],
            "end": row['end'],
            "label_6way": label_6way,
            "label_4way": LABEL_MAP_4WAY.get(label_6way, 3), # Default to Backg
            "label_2way": LABEL_MAP_2WAY.get(label_6way, 1), # Default to Backg
            "original_channel_idx": int(row['channel'])
        })

    return {
        "key": key,
        "samples": samples, # List of event dictionaries
        "classes_present": list(classes_present),
        "split": "train" if "train" in fpath else "eval",
        "channels": eeg_channels
    }

def build_lmdb(data_root, pre_root, configs):
    train_files = glob.glob(data_root + "/edf/train/*/*.edf")
    eval_files = glob.glob(data_root + "/edf/eval/*/*.edf")
    all_files = train_files + eval_files

    dfs = {}
    for cfg in configs:
        lmdb_path = os.path.join(pre_root, f"tuev_1s_{cfg['name']}")
        os.makedirs(lmdb_path, exist_ok=True)

        env = lmdb.open(lmdb_path, map_size=10995116277)
        metadata = list()
        channels = None

        with env.begin(write=True) as txn:
            for fpath in tqdm.tqdm(all_files):
                
                # Process file
                try:
                    outputs = process_single_file(fpath, cfg)
                except Exception as e:
                    print(f"Error processing {fpath}: {e}")
                    continue

                if not outputs["samples"]: continue # Skip files with no valid events
                if channels is None: channels = outputs["channels"]

                # We store the LIST of samples (events) associated with this file key
                # This keeps files grouped in LMDB, but we load specific events later
                txn.put(outputs["key"].encode("ascii"), pickle.dumps(outputs["samples"]))

                # Get stratification label (rarest event in this file)
                strat_label = get_stratification_label(outputs["classes_present"])

                metadata.append({
                    "key": outputs["key"],
                    "split": outputs["split"],
                    "num_events": len(outputs["samples"]), # How many events in this file
                    "strat_label": strat_label,
                    "patient_id": outputs["key"].split('_')[0]
                })
        
        env.close()

        # Save channels
        channels_path = os.path.join(lmdb_path, "channels.json")
        json.dump(channels, open(channels_path, "w"), indent=4)
        
        # Save Metadata
        df = pd.DataFrame(metadata)
        df.to_csv(os.path.join(pre_root, f"tuev_1s_{cfg['name']}_metadata.csv"), index=False)
        dfs[cfg['name']] = df
        
    return dfs

def generate_splits(df, split_path):
    print(f"\n---- Generating 5-Fold Splits ----")
    
    # 1. Separate Hold-out (Eval)
    train_full_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "eval"].reset_index(drop=True)
    
    # 2. Extract Groups (Patient IDs)
    groups = train_full_df["patient_id"].values
    
    # 3. Targets for Stratification
    y = train_full_df["strat_label"].values
    X = train_full_df["key"].values 

    # 4. StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    fold_idx = 0
    for train_idxs, val_idxs in sgkf.split(X, y, groups):
        
        fold_train_df = train_full_df.iloc[train_idxs]
        fold_val_df = train_full_df.iloc[val_idxs]
        
        def format_list(dataframe):
            return [{
                "key": row["key"],
                "split": row["split"],
                "length": int(row["num_events"]) # Use num_events for dataloader length
            } for _, row in dataframe.iterrows()]

        out_data = {
            "train": format_list(fold_train_df),
            "val": format_list(fold_val_df),
            "test": format_list(test_df)
        }
        
        filename = f"cv_fold{fold_idx}.json"
        with open(os.path.join(split_path, filename), "w") as f:
            json.dump(out_data, f, indent=4)
        
        print(f"Fold {fold_idx}: Train Files: {len(fold_train_df)}, Val Files: {len(fold_val_df)}")
        
        fold_idx += 1

if __name__ == "__main__":
    data_root, pre_root, split_path = get_paths()
    
    configs = [
        # {"name": "raw_norm_200", "fs": 200, "norm": True, "robust": False},
        {"name": "raw_robust_200", "fs": 200, "norm": False, "robust": True},
        # {"name": "raw_norm_5s_200", "fs": 200, "norm": True, 'duration': 5.0, "robust": False},
        # {"name": "raw_robust_5s_200", "fs": 200, "norm": False, 'duration': 5.0, "robust": True},
        # {'name': 'csbrain_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'csbrain_200', "robust": False},
        # {'name': 'csbrain_5s_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'csbrain_200', 'duration': 5.0, "robust": False},
        # {'name': 'labram_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'labram_200', "robust": False},
        # {'name': 'labram_5s_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'labram_200', 'duration': 5.0, "robust": False},
        # {'name': 'cbramod_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'cbramod_200'}
    ]

    dfs = build_lmdb(data_root=data_root, pre_root=pre_root, configs=configs)
    
    if "raw_norm_200" in dfs:
        generate_splits(dfs["raw_norm_200"], split_path)
