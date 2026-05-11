# TUEV dataset, 6 classes
# https://isip.piconepress.com/projects/nedc/html/tuh_eeg/#c_tuev
# This corpus is a subset of TUEG that contains annotations of EEG segments as one of six classes: (1) spike and sharp wave (SPSW), (2) generalized periodic epileptiform discharges (GPED), (3) periodic lateralized epileptiform discharges (PLED), (4) eye movement (EYEM), (5) artifact (ARTF) and (6) background (BCKG).

import os, sys, json
import mne 
import numpy as np
import pandas as pd 
import lmdb 
import pickle 
import glob
import tqdm

from sklearn.model_selection import train_test_split, StratifiedGroupKFold

from biodl.utils.make_ssl_data import process_ssl_data

dataset_name = "tuev"
RANDOM_STATE = 42 

TCP_MONTAGE = [
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "A1-T3",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "T4-A2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2"
]

SSL_MONTAGE = [
    "FP1",
    "F7",
    "T3",
    "T5",
    "O1",
    "FP2",
    "F8",
    "T4",
    "T6",
    "O2",
    # "A1", A1, A2 are ignored
    # "T3",
    "C3",
    "CZ",
    "C4",
    # "T4",
    # "A2",
    "F3",
    # "C3",
    "P3",
    "F4",
    # "C4",
    "P4",
    # "O2"
]

CLASS_PRIORITY = {
    2: 0,  # GPED (Most Critical)
    3: 1,  # PLED
    1: 2,  # SPSW
    4: 3,  # EYEM
    5: 4,  # ARTF
    6: 5   # BCKG (Least Critical)
}

def get_stratification_label(unique_labels_in_file):
    """Returns the single label ID that is 'rarest' or most important in the file."""
    if not unique_labels_in_file:
        return 6 # Default to Background
    
    # Sort labels by priority (lowest rank first)
    best_label = sorted(unique_labels_in_file, key=lambda x: CLASS_PRIORITY.get(x, 100))[0]
    return best_label

def convert_to_tcp_montage(raw):
    ch_names = raw.info["ch_names"]
    MODE = raw.info["ch_names"][0].split("-")[-1]

    eeg_data = raw.get_data(units="uV")
    tcp_eeg_data = np.zeros((len(TCP_MONTAGE), eeg_data.shape[-1]))

    for i, ch in enumerate(TCP_MONTAGE):
        ch_1, ch_2 = ch.split("-")
        tcp_eeg_data[i] = eeg_data[ch_names.index(f"EEG {ch_1}-{MODE}")] - eeg_data[ch_names.index(f"EEG {ch_2}-{MODE}")]
    
    # make the montage and raw data structure for converted data
    tcp_mne_info = mne.create_info(
        ch_names=TCP_MONTAGE,
        sfreq=raw.info["sfreq"],
        ch_types=["eeg"]*len(TCP_MONTAGE)
    )

    return mne.io.RawArray(tcp_eeg_data, tcp_mne_info)

def convert_to_ssl_montage(raw):
    ch_names = raw.info["ch_names"]
    MODE = raw.info["ch_names"][0].split("-")[-1]

    eeg_data = raw.get_data(units="uV")
    ssl_eeg_data = np.zeros((len(SSL_MONTAGE), eeg_data.shape[-1]))

    for i, ch in enumerate(SSL_MONTAGE):
        ssl_eeg_data[i] = eeg_data[ch_names.index(f"EEG {ch}-{MODE}")]
    
    ssl_mne_info = mne.create_info(
        ch_names = SSL_MONTAGE,
        sfreq=raw.info["sfreq"],
        ch_types =["eeg"] * len(SSL_MONTAGE)
    )

    return mne.io.RawArray(ssl_eeg_data, ssl_mne_info)

def get_paths():
    root = os.getenv("BIODL_RAW_DATA_ROOT", "./data")
    data_root = os.path.join(root, dataset_name) 
    
    pre_root = os.getenv("BIODL_PREPROCESSED_DATA_ROOT", "./data_preprocessed")
    split_path = os.path.join(os.getenv("BIODL_SPLIT_PATH", "./splits"), dataset_name)
    os.makedirs(split_path, exist_ok=True)
    
    return data_root, pre_root, split_path

def download_dataset(data_root):
    if os.path.exists(data_root):
        print(f"Skipping downloading dataset, already exists at {data_root} !")
    else:
        credentials_path = os.getenv("TUEG_SSH_KEY")
        os.system(f'rsync -auvxL -e "ssh -i {credentials_path}" nedc_tuh_eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_events/v2.0.1 {data_root}')
        print(f"Download Complete. Dataset stored at: {data_root}")

def process_single_file(fpath, config):

    raw = mne.io.read_raw_edf(fpath, preload=True)
    # In order to use the channel information in TUEV, we need to convert the montage to Bipolar first
    if "ssl" not in config.keys():
        raw_tcp = convert_to_tcp_montage(raw)
    else:
        raw_tcp = convert_to_ssl_montage(raw)
        # eeg_channels = [ch.split('-')[0].replace("EEG ") for ch in raw_tcp.info["ch_names"]]
    
    eeg_channels = raw_tcp.info["ch_names"]
    # notch filter
    raw_tcp.notch_filter(np.arange(60, raw_tcp.info['sfreq']/2, 60), verbose=False)

    # filter 
    raw_tcp.filter(l_freq=0.1, h_freq=75, verbose=False)

    # resample data
    if raw.info['sfreq'] != config["fs"]:
        raw_tcp.resample(config['fs'], verbose=False, method="polyphase", n_jobs=5)
    
    if len(raw_tcp.info['bads']) > 0:
                print('interpolate_bads')
                raw_tcp.interpolate_bads()
    
    data = raw_tcp.get_data(units="uV").astype(np.float32)

    if config.get('ssl', None) is not None:
        # print(f"Entered SSL loop: {cfg.get('ssl')}")
        data, eeg_channels = process_ssl_data(data, eeg_channels, ssl=config['ssl'])
        # if "channels" not in storage[cfg['name']]:
        #     storage[cfg['name']]['channels'] = out_channels

    # z-score norm
    if config["norm"]:
        mu = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        std[std == 0] = 1.0 # Avoid div/0
        data = (data - mu) / std
    
    # get the annotations to create the input samples
    metadata = list()
    key = os.path.basename(fpath).replace(".edf", "")

    annotations = pd.read_csv(fpath.replace('.edf', '.rec'), delimiter=',', header=None, comment='#')
    annotations.columns = ['channel', 'start', 'end', 'label'] if annotations.shape[1] == 4 else ['channel', 'start', 'end', 'label', 'conf']

    all_labels_in_file = annotations["label"].unique().tolist()
    all_labels_in_file = [int(l) for l in all_labels_in_file] # Ensure ints

    start_times = list(annotations["start"].unique())
    for s in start_times:
        df_s = annotations[annotations["start"] == s] # start time for dataframe
        end_time = list(df_s["end"].unique())[0]
        
        labels = sorted(list(df_s["label"].unique()))
                
        metadata.append({
            "dataframe": df_s,
            "label": labels[0],
            "start": s,
            "end": end_time,
            "labels": labels,
        })
    return {
        "data": data,
        "key": key,
        "metadata": metadata,
        "split": "train" if "train" in fpath else "eval",
        "length": len(metadata),
        "classes_present": all_labels_in_file,
        "channels": eeg_channels
    }

def build_lmdb(data_root, pre_root, configs):
    train_files = glob.glob(data_root + "/edf/train/*/*.edf")
    eval_files = glob.glob(data_root + "/edf/eval/*/*.edf")
    all_files = train_files + eval_files

    dfs = {}
    for cfg in configs:
        lmdb_path = os.path.join(pre_root, f"{dataset_name}_{cfg['name']}")
        os.makedirs(lmdb_path, exist_ok=True)

        env = lmdb.open(lmdb_path, map_size=10995116277)
        metadata = list()
        channels = None

        with env.begin(write=True) as txn:
            for fpath in tqdm.tqdm(all_files):

                outputs = process_single_file(fpath, cfg)
                if channels is None: channels = outputs["channels"]
                txn.put(outputs["key"].encode("ascii"), pickle.dumps({
                    "data": outputs["data"],
                    "metadata": outputs["metadata"]
                }))

                strat_label = get_stratification_label(outputs["classes_present"])

                metadata.append({
                    "key": outputs["key"],
                    "split": outputs["split"],
                    "length": outputs["length"],
                    "strat_label": strat_label
                })
        
        env.close()

        # write the channels list
        channels_path = os.path.join(lmdb_path, "channels.json")
        json.dump(channels, open(channels_path, "w"), indent=4)
        
        df = pd.DataFrame(metadata)
        df.to_csv(os.path.join(pre_root, f"{dataset_name}_{cfg['name']}_metadata.csv"), index=False)
        dfs[cfg['name']] = df
    return dfs

def generate_splits(df, split_path):
    print(f"\n---- Generating 5-Fold Splits ----")
    
    # 1. Separate the official Hold-out (Eval) set
    # In TUEV, "eval" folder is the official test set. We DO NOT touch it for CV.
    train_full_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "eval"].reset_index(drop=True)
    
    # 2. Extract Groups (Patient IDs)
    # TUEV keys are usually: 'aaaaaadb_s004_t000' -> Patient is 'aaaaaadb'
    groups = train_full_df["key"].apply(lambda x: x.split("_")[0]).values
    
    # 3. Extract Targets for Stratification
    # We use the 'strat_label' we computed earlier (the rarest class in the file)
    y = train_full_df["strat_label"].values
    X = train_full_df["key"].values # Feature is just the key/index

    # 4. Initialize StratifiedGroupKFold
    # This ensures:
    #   a. No patient (group) appears in both Train and Val (Leakage prevention)
    #   b. The distribution of 'y' (classes) is roughly consistent across folds
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    fold_idx = 0
    for train_idxs, val_idxs in sgkf.split(X, y, groups):
        
        fold_train_df = train_full_df.iloc[train_idxs]
        fold_val_df = train_full_df.iloc[val_idxs]
        
        # Helper to format list for JSON
        def format_list(dataframe):
            return [{
                "key": row["key"],
                "split": row["split"],
                "length": int(row["length"]) # ensure int for JSON
            } for _, row in dataframe.iterrows()]

        out_data = {
            "train": format_list(fold_train_df),
            "val": format_list(fold_val_df),
            "test": format_list(test_df)
        }
        
        # 5. Save Fold JSON
        filename = f"cv_fold{fold_idx}.json"
        with open(os.path.join(split_path, filename), "w") as f:
            json.dump(out_data, f, indent=4)
        
        # Verification Print
        print(f"Fold {fold_idx}: Train Files: {len(fold_train_df)}, Val Files: {len(fold_val_df)}")
        print(f"   - Unique Patients Train: {fold_train_df['key'].apply(lambda x: x.split('_')[0]).nunique()}")
        print(f"   - Unique Patients Val:   {fold_val_df['key'].apply(lambda x: x.split('_')[0]).nunique()}")
        # Check overlap
        train_pats = set(fold_train_df['key'].apply(lambda x: x.split('_')[0]))
        val_pats = set(fold_val_df['key'].apply(lambda x: x.split('_')[0]))
        print(f"   - Patient Overlap: {len(train_pats.intersection(val_pats))} (Should be 0)")
        
        fold_idx += 1

if __name__ == "__main__":
    data_root, pre_root, split_path = get_paths()
    
    # download dataset
    download_dataset(data_root=data_root)

    configs = [
        {"name": "raw_norm_200", "fs": 200, "norm": True},
        # {'name': 'csbrain_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'csbrain_200'},
        # {'name': 'labram_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'labram_200'},
        # {'name': 'cbramod_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'cbramod_200'}
    ]

    # build LMDB preporcessing
    dfs = build_lmdb(data_root=data_root, pre_root=pre_root, configs=configs)

    # generate splits
    generate_splits(dfs["raw_norm_200"], split_path)