# Dataset for https://figshare.com/articles/dataset/EEG_Data_New/4244171 dataset
# Consists of resting state and task (P 300) data
# https://scholar.google.com/scholar?cluster=8556220843927428512&hl=en&oi=scholarr

# The dataset will have two modes
# 1. Load the Eyes closed/Eyes open resting state data
# 2. Use the task based EEG data

import os, sys, json
import mne 
import numpy as np
import pandas as pd 
import glob
import tqdm
import lmdb
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold

from biodl.utils.make_ssl_data import process_ssl_data

dataset_name = "mdd_mal"
RANDOM_STATE= 42
SAMPLING_RATE = None

EEG_CHANNELS = [
    'EEG Fp1-LE',
    'EEG F3-LE',
    'EEG C3-LE',
    'EEG P3-LE',
    'EEG O1-LE',
    'EEG F7-LE',
    'EEG T3-LE',
    'EEG T5-LE',
    'EEG Fz-LE',
    'EEG Fp2-LE',
    'EEG F4-LE',
    'EEG C4-LE',
    'EEG P4-LE',
    'EEG O2-LE',
    'EEG F8-LE',
    'EEG T4-LE',
    'EEG T6-LE',
    'EEG Cz-LE',
    'EEG Pz-LE',
]

def get_paths():

    data_root = os.path.join(os.getenv("BIODL_RAW_DATA_ROOT", "./"), dataset_name)

    pre_root = os.path.join(os.getenv("BIODL_PREPROCESSED_DATA_ROOT", "./preprocessed"))
    split_path = os.path.join(os.getenv("BIODL_SPLIT_PATH", "./splits"), dataset_name)
    os.makedirs(split_path, exist_ok=True)

    return data_root, pre_root, split_path

def download_dataset(data_root):

    if os.path.exists(data_root):
        print(f"Dataset already downloaded. Skipping downloading dataset!")
    else:
        os.makedirs(data_root, exist_ok=True)
        os.system(f"wget -O {data_root}/data.zip https://ndownloader.figshare.com/articles/4244171/versions/2")
        os.system(f"unzip {data_root}/data.zip -d {data_root}")        
        print(f"Downloaded dataset at {data_root}!")

def process_single_file(file_path, config):

    try:
        raw = mne.io.read_raw_edf(file_path, preload=True)
        raw.pick(EEG_CHANNELS)
        raw.set_eeg_reference(ref_channels='average')

        # resample data
        if raw.info["sfreq"] != config["fs"]:
            raw.resample(config["fs"], verbose=False, n_jobs=5, method="polyphase")

        # bandpass filter
        raw.filter(l_freq=0.5, h_freq=75, verbose=False)

        raw.notch_filter(np.arange(50, config["fs"]//2, 50), verbose=False)

        if len(raw.info['bads']) > 0:
                print('interpolate_bads')
                raw.interpolate_bads()

        data = raw.get_data(units="uV").astype(np.float32)

        if config.get('ssl', None) is not None:
            # print(f"Entered SSL loop: {cfg.get('ssl')}")
            forward_kwargs = {}
            forward_kwargs["input_channels"] = [ch.replace("EEG ", "").replace("-LE", "") for ch in EEG_CHANNELS]
            forward_kwargs["ssl"] = config['ssl']
            if config.get("norm_type", None): forward_kwargs["norm_type"] = config["norm_type"]

            data, eeg_channels = process_ssl_data(data, **forward_kwargs)
        else:
            eeg_channels = EEG_CHANNELS
        
        # z-score normalization
        if config["norm"]:
            mu = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            std[std == 0] = 1.0 # Avoid div/0
            data = (data - mu) / std

        return data, eeg_channels

    except Exception as e:
        print(f"Unable to process file: {file_path}: {e}")
        return None

def build_lmdb(data_root, pre_root, configs):
    eo_files = glob.glob(data_root + "/*EO*.edf")
    ec_files = glob.glob(data_root + "/*EC*.edf")

    all_files = eo_files + ec_files

    dfs = {}

    for cfg in configs:
        lmdb_prefix = f"{dataset_name}_{cfg['name']}"
        if "norm_type" in cfg.keys(): lmdb_prefix += f"_{cfg['norm_type']}"

        lmdb_path = os.path.join(pre_root, lmdb_prefix)
        os.makedirs(lmdb_path, exist_ok=True)

        env = lmdb.open(lmdb_path, map_size=10995116277)
        meta_data = list()

        with env.begin(write=True) as txn:
            for fpath in tqdm.tqdm(all_files):

                filename = os.path.basename(fpath)
                parts = filename.replace(".edf", "").split(" ")
                label = 0 if "H" in parts[0] else 1
                subject_id = parts[1]
                task_type = parts[2]
                key_str = f"{subject_id}_{task_type}_{label}"
                data, eeg_channels = process_single_file(fpath, cfg)
                if data is None: continue

                txn.put(key_str.encode("ascii"), pickle.dumps({
                    "data": data,
                    "label": label
                }))

                meta_data.append({
                    "key": key_str,
                    "label": label,
                    "samples": data["data"].shape[-1] if type(data) == dict else data.shape[-1],
                    "task": task_type,
                    "subject": f"{subject_id}_{label}"
                })
        
        env.close()

        channels = [ch.replace("EEG ", "").replace("-LE", "") for ch in eeg_channels]
        json.dump(channels, open(os.path.join(lmdb_path, "channels.json"), "w"), indent=4)
        # save metadata
        df = pd.DataFrame(meta_data)
        df.to_csv(os.path.join(pre_root, f"{dataset_name}_{cfg['name']}_metadata.csv"), index=False)
        dfs[cfg['name']] = df
    
    return dfs

def generate_splits(df, split_path):
    print(f"\n---- Generating Splits ----")
    
    def save_json(split_dict, filename):
        out_data = {}
        for phase, sub_df in split_dict.items():
            phase_list = []
            for _, row in sub_df.iterrows():
                phase_list.append({
                    "key": row['key'],
                    "label": int(row['label']),
                    "length": int(row['samples']),
                    "subject": row['subject'],
                    "task": row['task']
                })
            out_data[phase] = phase_list
        
        with open(os.path.join(split_path, filename), 'w') as f:
            json.dump(out_data, f, indent=4)
        print(f"Saved {filename}")

    # Define the three scenarios: EO only, EC only, and Mixed
    scenarios = [
        {"name": "EO", "mask": df["task"] == "EO"},
        {"name": "EC", "mask": df["task"] == "EC"},
        {"name": "mixed", "mask": np.ones(len(df), dtype=bool)} # Selects all
    ]

    for scen in scenarios:
        task_mode = scen["name"]
        # Filter DataFrame based on the scenario
        current_df = df[scen["mask"]].copy()

        print(f"Processing Scenario: {task_mode} ({len(current_df)} samples)")

        # --- Subject Level Stratification ---
        # We must split by Subject ID to prevent data leakage.
        # We also need the label (Healthy vs MDD) for each subject to perform Stratified K-Fold.
        # Group by 'subject' and take the first label found (subject label is constant).
        subject_meta = current_df.groupby("subject")["label"].first().reset_index()
        
        unique_subjects = subject_meta["subject"].values
        subject_labels = subject_meta["label"].values

        # 5-Fold Cross Validation
        # StratifiedKFold ensures we have a balanced ratio of Healthy/MDD in every fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(unique_subjects, subject_labels)):
            
            # Map indices back to Subject IDs
            train_val_subs = unique_subjects[train_val_idx]
            test_subs = unique_subjects[test_idx]
            
            # --- Internal Train/Val Split ---
            # Split the training pool into Train (80%) and Val (20%)
            # We assume subject_labels aligns with unique_subjects indices
            train_val_labels = subject_labels[train_val_idx]

            try:
                train_subs, val_subs = train_test_split(
                    train_val_subs, 
                    test_size=0.2, 
                    stratify=train_val_labels, # Attempt to maintain class balance in Val
                    random_state=RANDOM_STATE
                )
            except ValueError:
                # Fallback if a class has too few members for stratification in the inner loop
                print(f"Warning: Insufficient classes for stratification in fold {fold_idx}. Using random split.")
                train_subs, val_subs = train_test_split(
                    train_val_subs, 
                    test_size=0.2, 
                    random_state=RANDOM_STATE
                )

            # Reconstruct the DataFrames based on Subject Lists
            split_dict = {
                "train": current_df[current_df["subject"].isin(train_subs)],
                "val": current_df[current_df["subject"].isin(val_subs)],
                "test": current_df[current_df["subject"].isin(test_subs)]
            }

            # Filename example: mdd_mal_EO_fold0.json
            save_json(split_dict, f"{dataset_name}_{task_mode}_fold{fold_idx}.json")

if __name__ == "__main__":
    data_root, pre_root, split_path = get_paths()
    print(f"Downloading MDD MAL dataset")
    download_dataset(data_root=data_root)

    # 2. Process
    configs = [
        # {'name': 'raw_norm_256', 'fs': 256, 'norm': True},
        # {'name': 'raw_norm_200', 'fs': 200, 'norm': True}
        # {'name': 'csbrain_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'csbrain_200'},
        # {'name': 'labram_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'labram_200'},
        # {'name': 'cbramod_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'cbramod_200'}
        {'name': 'eegfm_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'eegfm', 'norm_type': 'uv_norm'}
    ]
    meta_df = build_lmdb(data_root, pre_root, configs)
    
    # 3. Splits
    for cfg in configs:
        meta_df_cfg = meta_df[cfg['name']]
        if meta_df_cfg is not None:
            generate_splits(meta_df_cfg, split_path)


