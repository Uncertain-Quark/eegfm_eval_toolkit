import os
import json
import shutil
import mne
import numpy as np
import tqdm
from scipy import signal
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from biodl.utils.make_ssl_data import process_ssl_data

# --- Configuration ---
RANDOM_STATE = 42
N_FOLDS = 5
EVENT_ID_DICT = {"T1": 0, "T2": 1}
EVENT_START_SEC = 0
EVENT_END_SEC = 4
IGNORED_SUBJECTS = [88, 90, 92, 100]
ROUNDS_LR = [3, 4, 7, 8, 11, 12]
ROUNDS_FF = [5, 6, 9, 10, 13, 14]

selected_channels = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..',
                     'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.',
                     'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..',
                     'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..',
                     'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.',
                     'O1..', 'Oz..', 'O2..', 'Iz..']

def parse_keys_to_df(subject_to_keys):
    """
    Flattens the subject_to_keys dictionary into a Pandas DataFrame.
    Key Format assumed: S{subject}_C{class}_{run}_{trial}
    """
    all_keys = []
    for sub, keys in subject_to_keys.items():
        all_keys.extend(keys)
        
    data = []
    for k in all_keys:
        # k looks like: "S001_C0_03_001"
        parts = k.split('_')
        sub_id = int(parts[0][1:]) # S001 -> 1
        class_id = int(parts[1][1:]) # C0 -> 0
        run_id = int(parts[2]) # 03 -> 3
        
        data.append({
            "key": k,
            "subject": sub_id,
            "class": class_id,
            "run": run_id
        })
        
    df = pd.DataFrame(data)
    
    # Add High-Level columns for easier filtering
    # Real vs Imagined Mapping
    # Real: 3, 7, 11 (LR) | 5, 9, 13 (FF)
    # Imagined: 4, 8, 12 (LR) | 6, 10, 14 (FF)
    real_runs = [3, 7, 11, 5, 9, 13]
    imag_runs = [4, 8, 12, 6, 10, 14]
    
    # Task Type Mapping
    # LR (Left/Right): 3, 4, 7, 8, 11, 12
    # FF (Fists/Feet): 5, 6, 9, 10, 13, 14
    lr_runs = [3, 4, 7, 8, 11, 12]
    
    df['is_real'] = df['run'].isin(real_runs)
    df['is_lr_task'] = df['run'].isin(lr_runs)
    
    return df

def get_paths(dataset_name="mmiphysionet"):
    root = os.getenv("BIODL_RAW_DATA_ROOT", "./data")
    data_root = os.path.join(root, dataset_name)
    
    pre_root = os.getenv("BIODL_PREPROCESSED_DATA_ROOT", "./data_preprocessed")
    split_path = os.path.join(os.getenv("BIODL_SPLIT_PATH", "./splits"), dataset_name)
    os.makedirs(split_path, exist_ok=True)
    
    return data_root, pre_root, split_path

def compute_spectrogram(data, fs):
    f, t, Zxx = signal.stft(data, fs=fs, nperseg=fs//2) 
    return np.abs(Zxx).astype(np.float32)

def download_dataset(data_root):
    if not os.path.exists(data_root):
        print(f"Downloading dataset to {data_root}...")
        os.makedirs(data_root, exist_ok=True)
        os.system(f"aws s3 sync --no-sign-request s3://physionet-open/eegmmidb/1.0.0/ {data_root}")

def process_and_store_memory(data_root, pre_root, configs, l_freq=0.3, h_freq=None, dataset_name: str="mmiphysionet"):
    """
    Accumulates all data in RAM and saves as .npz files.
    """
    subjects = [i for i in range(1, 110) if i not in IGNORED_SUBJECTS]
    # subjects = [i for i in range(1, 110)]
    
    # Initialize storage containers for each config
    # Structure: {'raw_norm': {'X': [], 'y': [], 'keys': []}, ...}
    storage = {cfg['name']: {'X': [], 'y': [], 'keys': []} for cfg in configs}

    print("Starting In-Memory Processing...")
    
    # We track keys to ensure we can generate splits later
    subject_to_keys = {}

    for subject_id in tqdm.tqdm(subjects, desc="Subjects"):
        subject_keys = []
        subject_dir = os.path.join(data_root, f"S{subject_id:03d}")
        
        # Identify valid files
        all_rounds = ROUNDS_LR + ROUNDS_FF
        valid_files = []
        for r in all_rounds:
            fpath = os.path.join(subject_dir, f"S{subject_id:03d}R{r:02d}.edf")
            if os.path.exists(fpath): valid_files.append((r, fpath))
            
        for run_id, edf_file in valid_files:
            try:
                raw_base = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
                events_base, _ = mne.events_from_annotations(raw_base, event_id=EVENT_ID_DICT, verbose=False)
            except: continue

            raw_base.pick(selected_channels)
            if len(raw_base.info['bads']) > 0:
                print('interpolate_bads')
                raw_base.interpolate_bads()
            raw_base.set_eeg_reference(ref_channels='average')

            # store the channel order in storage dict
            ch_order = [ch.lower().strip(".") for ch in raw_base.info["ch_names"]]
            # storage["channels"] = ch_order
            
            # raw_base.filter(0.3, 50, method="fir", phase="zero", verbose=False)
            raw_base.filter(l_freq, h_freq, method="fir", phase="zero", verbose=False)
            # Pre-filtering
            raw_base.notch_filter(60, method="iir", phase="zero", verbose=False)
            

            # --- Resample Caching ---
            unique_rates = set(c['fs'] for c in configs)
            resampled_cache = {}
            for fs in unique_rates:
                if int(raw_base.info["sfreq"]) != fs:
                    r_tmp, e_tmp = raw_base.copy().resample(fs, events=events_base, method="polyphase", n_jobs=8)
                    resampled_cache[fs] = (r_tmp, e_tmp)
                else:
                    resampled_cache[fs] = (raw_base, events_base)

            # --- Process Trials ---
            # Use the first fs config as the "reference" for trial counting
            ref_fs = list(unique_rates)[0]
            _, ref_events = resampled_cache[ref_fs]
            
            label_offset = 2 if run_id in ROUNDS_FF else 0
            n_trials = len(ref_events)

            for i in range(n_trials):
                # Unique Key for Metadata
                ref_lbl = ref_events[i][2] + label_offset
                key_str = f"S{subject_id:03d}_C{ref_lbl}_{run_id:02d}_{i:03d}"
                subject_keys.append(key_str)

                # Append data for each config
                for cfg in configs:
                    fs = cfg['fs']
                    
                    raw_res, events_res = resampled_cache[fs]
                    start_sample = events_res[i][0]
                    event_end_sec = EVENT_END_SEC if cfg.get("EVENT_END_SEC", None) is None else cfg["EVENT_END_SEC"]

                    start_idx = start_sample + int(EVENT_START_SEC * fs)
                    end_idx = start_sample + int(event_end_sec * fs)
                    
                    data_chunk = raw_res.get_data(units="uV")[:, start_idx:end_idx]
                    
                    # Size check
                    expected_len = int((event_end_sec - EVENT_START_SEC) * fs)
                    if data_chunk.shape[1] != expected_len: continue 

                    if cfg.get('ssl', None) is not None:
                        # print(f"Entered SSL loop: {cfg.get('ssl')}")
                        processed_out, out_channels = process_ssl_data(data_chunk, ch_order, ssl=cfg['ssl'])

                        if isinstance(processed_out, dict):
                            data_chunk = processed_out["data"]
                            # Store channel_idx globally for this config (it's the same for all trials)
                            if "channel_idx" not in storage[cfg['name']] and "channel_idx" in processed_out:
                                storage[cfg['name']]['channel_idx'] = processed_out["channel_idx"]
                        else:
                            data_chunk = processed_out

                        if "channels" not in storage[cfg['name']]:
                            storage[cfg['name']]['channels'] = out_channels

                    # Normalization
                    if cfg['norm']:
                        # Global Z-score per trial (simplest for this architecture)
                        mu = np.mean(data_chunk, axis=1, keepdims=True)
                        std = np.std(data_chunk, axis=1, keepdims=True)
                        std[std==0] = 1.0
                        data_chunk = (data_chunk - mu) / std

                    # Features
                    if cfg.get('spec', False):
                        final_data = compute_spectrogram(data_chunk, fs)
                    else:
                        final_data = data_chunk.astype(np.float32)

                    # Store in list
                    storage[cfg['name']]['X'].append(final_data)
                    storage[cfg['name']]['y'].append(ref_lbl)
                    storage[cfg['name']]['keys'].append(key_str)

                    if "channels" not in storage[cfg['name']]:
                        storage[cfg['name']]['channels'] = ch_order

        subject_to_keys[subject_id] = subject_keys
    # import pdb; pdb.set_trace()
    # --- SAVE TO DISK ---
    print("Saving monolithic arrays to disk...")
    for cfg_name, data_dict in storage.items():
        if cfg_name == "channels": continue
        # Stack lists into numpy arrays
        X_all = np.stack(data_dict['X'])
        y_all = np.array(data_dict['y'])
        keys_all = np.array(data_dict['keys']) # Save keys to map back to splits

        save_dir = os.path.join(pre_root, f"{dataset_name}_{cfg_name}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "data.npz")
        
        print(f"Saving {cfg_name} to {save_path} | Shape: {X_all.shape}")
        
        # # Save compressed to save disk space, or uncompressed (.save) for faster write
        # np.savez(save_path, X=X_all, y=y_all, keys=keys_all)
        save_kwargs = {'X': X_all, 'y': y_all, 'keys': keys_all}
        if 'channel_idx' in data_dict:
            save_kwargs['eegfm_channel_idx'] = np.array(data_dict['channel_idx'])
            
        np.savez(save_path, **save_kwargs)

        ch_path = os.path.join(save_dir, "channels.json")
        print(f"Storing channel configuration of {cfg_name} to {ch_path}")
        json.dump(storage[cfg['name']]["channels"], open(ch_path, "w"), indent=4)
        
    return subject_to_keys

def generate_all_splits(subject_to_keys, split_path):
    print("Parsing Metadata from keys...")
    df = parse_keys_to_df(subject_to_keys)
    subjects = sorted(df['subject'].unique())
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Helper to save json
    def save_split(name, train_df, val_df, test_df):
        out = {
            "train": train_df['key'].tolist(),
            "val": val_df['key'].tolist(),
            "test": test_df['key'].tolist()
        }
        path = os.path.join(split_path, f"{name}.json")
        with open(path, 'w') as f:
            json.dump(out, f, indent=4)
        print(f"  Saved: {name} (Train: {len(train_df)}, Test: {len(test_df)})")

    print(f"Generating splits in {split_path}...")

    # ==========================================
    # 1. CBraMod Split 
    # 0-70 in train, 70-89 in val and 89-109 in test
    # ==========================================
    cbramod_train = subjects[:70]
    cbramod_val = subjects[70:89]
    cbramod_test = subjects[89:109]

    cbramod_train_df = df[df['subject'].isin(cbramod_train)]
    cbramod_val_df = df[df['subject'].isin(cbramod_val)]
    cbramod_test_df = df[df['subject'].isin(cbramod_test)]

    save_split(f"4class_cbramod_split_fold0", cbramod_train_df, cbramod_val_df, cbramod_test_df)

    # ==========================================
    # 1. 4-Class Splits (Cross-Subject)
    # ==========================================
    print("\n--- Generating 4-Class Cross-Subject ---")
    # Use all data
    for fold, (train_idx, test_idx) in enumerate(kf.split(subjects)):
        train_subs = [subjects[i] for i in train_idx]
        test_subs = [subjects[i] for i in test_idx]
        
        # Split Train into Train/Val (subjects)
        tr_subs, val_subs = train_test_split(train_subs, test_size=0.2, random_state=42)
        
        train_df = df[df['subject'].isin(tr_subs)]
        val_df = df[df['subject'].isin(val_subs)]
        test_df = df[df['subject'].isin(test_subs)]
        
        save_split(f"4class_cross_subject_fold{fold}", train_df, val_df, test_df)

    # ==========================================
    # 2. 4-Class Splits (Within-Subject)
    # ==========================================
    print("\n--- Generating 4-Class Within-Subject ---")
    # Train: First 2 blocks of each type (LR: 3,4,7,8 | FF: 5,6,9,10)
    # Test:  Last 1 block of each type  (LR: 11,12   | FF: 13,14)
    train_runs = [3, 4, 7, 8, 5, 6, 9, 10]
    test_runs = [11, 12, 13, 14]
    
    # Filter whole dataframe
    ws_train = df[df['run'].isin(train_runs)]
    ws_test = df[df['run'].isin(test_runs)]
    
    # Create simple train/val split from the training runs
    ws_tr_final, ws_val_final = train_test_split(ws_train, test_size=0.2, random_state=42, stratify=ws_train['subject'])
    
    # We only need one file for within-subject usually, but we can make folds if needed.
    # Here we just make one "Fold 0" as it's a fixed split definition.
    save_split("4class_within_subject_fold0", ws_tr_final, ws_val_final, ws_test)

    # ==========================================
    # 3. 2-Class (Left vs Right) Cross-Subject
    # ==========================================
    print("\n--- Generating 2-Class (Left/Right) Cross-Subject ---")
    # Filter: Only LR runs
    df_lr = df[df['is_lr_task']].copy()
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(subjects)):
        train_subs = [subjects[i] for i in train_idx]
        test_subs = [subjects[i] for i in test_idx]
        
        tr_subs, val_subs = train_test_split(train_subs, test_size=0.2, random_state=42)
        
        train_df = df_lr[df_lr['subject'].isin(tr_subs)]
        val_df = df_lr[df_lr['subject'].isin(val_subs)]
        test_df = df_lr[df_lr['subject'].isin(test_subs)]
        
        save_split(f"2class_left_right_cross_subject_fold{fold}", train_df, val_df, test_df)

    # ==========================================
    # 4. 2-Class (Left vs Right) Within-Subject
    # ==========================================
    print("\n--- Generating 2-Class (Left/Right) Within-Subject ---")
    # Use only LR runs.
    # Train: 3, 4, 7, 8
    # Test: 11, 12
    lr_train_runs = [3, 4, 7, 8]
    lr_test_runs = [11, 12]
    
    df_lr_ws = df[df['is_lr_task']] # Ensure we don't accidentally get feet data
    
    ws_lr_train = df_lr_ws[df_lr_ws['run'].isin(lr_train_runs)]
    ws_lr_test = df_lr_ws[df_lr_ws['run'].isin(lr_test_runs)]
    
    ws_lr_tr_final, ws_lr_val_final = train_test_split(ws_lr_train, test_size=0.2, random_state=42, stratify=ws_lr_train['subject'])
    
    save_split("2class_left_right_within_subject_fold0", ws_lr_tr_final, ws_lr_val_final, ws_lr_test)

    # ==========================================
    # 5. Real vs Imagined (Transfer Learning)
    # ==========================================
    print("\n--- Generating Real vs Imagined (All Classes) ---")
    # Train on Real, Test on Imagined
    
    real_df = df[df['is_real']]
    imag_df = df[~df['is_real']]
    
    # Simple Split: Train on Real, Test on Imagined
    # We split Real into Train/Val
    rvs_train, rvs_val = train_test_split(real_df, test_size=0.2, random_state=42, stratify=real_df['subject'])
    
    save_split("real_vs_imagined_all_subjects_fold0", rvs_train, rvs_val, imag_df)
    
    # Note: If you want 2-Class Real vs Imagined (only LR), just filter `df_lr` instead of `df`
    print("\n--- Generating Real vs Imagined (Left/Right Only) ---")
    real_lr_df = df_lr[df_lr['is_real']]
    imag_lr_df = df_lr[~df_lr['is_real']]
    
    rvs_lr_train, rvs_lr_val = train_test_split(real_lr_df, test_size=0.2, random_state=42, stratify=real_lr_df['subject'])
    save_split("real_vs_imagined_left_right_fold0", rvs_lr_train, rvs_lr_val, imag_lr_df)

    print("\n--- Generating Motor vs Subject ID Tasks (Subset of 4 Subjects) ---")
    
    # Grab the first 4 available subjects
    # (Since IGNORED_SUBJECTS filters out bad ones, subjects[:4] is safe)
    subset_subjects = subjects[:4] 
    df_subset = df[df['subject'].isin(subset_subjects)].copy()
    
    # Define the runs as requested
    subset_train_runs = [3, 4, 7, 8, 5, 6, 9, 10]
    subset_test_runs = [11, 12, 13, 14]
    
    # Filter the dataframe by the specific runs
    subset_train_pool = df_subset[df_subset['run'].isin(subset_train_runs)]
    subset_test_df = df_subset[df_subset['run'].isin(subset_test_runs)]
    
    # Split the training pool into 80% train / 20% validation
    # We stratify by both 'subject' and 'class' to ensure the 20% val set is perfectly balanced 
    # for both the Motor classification task and the Subject ID classification task.
    subset_train_df, subset_val_df = train_test_split(
        subset_train_pool, 
        test_size=0.2, 
        random_state=42, 
        stratify=subset_train_pool[['subject', 'class']]
    )
    
    # Save the identically structured split files for your two different downstream tasks
    save_split("4class_within_subject_motor_fold0", subset_train_df, subset_val_df, subset_test_df)
    save_split("4class_within_subject_id_fold0", subset_train_df, subset_val_df, subset_test_df)

if __name__ == "__main__":
    import sys 
    event_end_sec = int(sys.argv[1]) # 4s
    
    dataset_name = "mmiphysionet"
    data_root, pre_root, split_path = get_paths(dataset_name=dataset_name)
    download_dataset(data_root)

    processing_configs = [
        # {'name': f'raw_norm_{event_end_sec}_160', 'fs': 160, 'norm': True, 'spec': False, 'EVENT_END_SEC': event_end_sec},
        # {'name': f'raw_norm_{event_end_sec}_200', 'fs': 200, 'norm': True, 'spec': False, 'EVENT_END_SEC': event_end_sec},
        {
            "name": f"eegfm_montage_{event_end_sec}_200", "fs": 200, "norm": False, "spec": False, "ssl": "eegfm", "norm_type": "uv_norm", "EVENT_END_SEC": event_end_sec
        }
        # {'name': f'cbramod_{event_end_sec}_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'cbramod_200', 'EVENT_END_SEC': event_end_sec},
        # {'name': f'labram_{event_end_sec}_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'labram_200', 'EVENT_END_SEC': event_end_sec},
        # {'name': f'csbrain_{event_end_sec}_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'csbrain_200', 'EVENT_END_SEC': event_end_sec}
    ]

    keys_map = process_and_store_memory(data_root, pre_root, processing_configs, dataset_name=dataset_name)
    generate_all_splits(keys_map, split_path)