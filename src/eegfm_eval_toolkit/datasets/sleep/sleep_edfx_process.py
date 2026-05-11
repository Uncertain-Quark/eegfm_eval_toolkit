import os
import json
import lmdb
import pickle
import mne
import numpy as np
import tqdm
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split, KFold
from concurrent.futures import ProcessPoolExecutor, as_completed

from biodl.utils.make_ssl_data import process_ssl_data

# --- Configuration ---
dataset_name = "sleep_edfx"
RANDOM_STATE = 42

TARGET_CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']

STAGE_MAP = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,  # Merge N4 into N3
    'Sleep stage R': 4,
    'Movement time': -1,
    'Sleep stage ?': -1,
}


def get_paths():
    root = os.getenv("BIODL_RAW_DATA_ROOT", "./data")
    data_root = os.path.join(root, dataset_name)
    pre_root = os.getenv("BIODL_PREPROCESSED_DATA_ROOT", "./data_preprocessed")
    split_path = os.path.join(os.getenv("BIODL_SPLIT_PATH", "./splits"), dataset_name)
    os.makedirs(split_path, exist_ok=True)
    return data_root, pre_root, split_path


def download_dataset(data_root):
    import shutil
    if shutil.which("aws") is None:
        raise RuntimeError("AWS CLI is not installed. Run: pip install awscli")

    if os.path.exists(data_root) and len(os.listdir(data_root)) > 0:
        print(f"Skipping download. Data already exists at: {data_root}")
        return

    print(f"Downloading Sleep-EDFx to: {data_root}")
    os.makedirs(data_root, exist_ok=True)
    cmd = f"aws s3 sync --no-sign-request s3://physionet-open/sleep-edfx/1.0.0/ {data_root}"
    if os.system(cmd) != 0:
        raise RuntimeError("Download failed. Check your internet connection or AWS CLI.")
    print("Download complete.")


def _process_single_pair(args):
    """
    Top-level worker function: loads one PSG+Hypnogram pair for all configs.
    Returns (rec_id, metadata_dict, {cfg_name: (data, labels, channels)}) or None on failure.
    """
    psg_path, hyp_path, configs = args

    fname = os.path.basename(psg_path)
    rec_id = fname.replace("-PSG.edf", "")
    prefix = fname[:2]
    subject_id = fname[3:5]
    night_id = fname[5]

    try:
        raw_base = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
        annot = mne.read_annotations(hyp_path)
        annot.crop(annot[0]['onset'], annot[-1]['onset'])
        raw_base.set_annotations(annot, emit_warning=False)
    except Exception as e:
        print(f"[Error] Failed to read {fname}: {e}")
        return None

    available = raw_base.ch_names
    if any(ch not in available for ch in TARGET_CHANNELS):
        print(f"[Warning] Missing target channels in {fname}, skipping.")
        return None

    raw_base.pick(TARGET_CHANNELS)
    if raw_base.info['bads']:
        raw_base.interpolate_bads()

    cfg_results = {}
    for cfg in configs:
        raw = raw_base.copy()

        if int(raw.info['sfreq']) != cfg['fs']:
            raw.resample(cfg['fs'], verbose=False, method="polyphase", n_jobs=4)

        raw.filter(l_freq=0.3, h_freq=35, verbose=False)

        events, _ = mne.events_from_annotations(raw, event_id=STAGE_MAP, chunk_duration=30., verbose=False)
        events = events[events[:, 2] != -1]

        if len(events) == 0:
            continue

        tmax = 30. - 1.0 / raw.info['sfreq']
        epochs = mne.Epochs(raw, events, event_id=None, tmin=0., tmax=tmax,
                            baseline=None, verbose=False, on_missing='ignore')

        data = epochs.get_data(units="uV").astype(np.float32)
        labels = events[:, 2]

        if len(data) == 0:
            continue

        if cfg.get('ssl') is not None:
            data, eeg_channels = process_ssl_data(
                data, [ch.replace("EEG ", "") for ch in TARGET_CHANNELS], ssl=cfg['ssl']
            )
        else:
            eeg_channels = [ch.replace("EEG ", "") for ch in TARGET_CHANNELS]

        if cfg['norm']:
            mu = np.mean(data, axis=2, keepdims=True)
            std = np.std(data, axis=2, keepdims=True)
            std[std == 0] = 1.0
            data = (data - mu) / std

        cfg_results[cfg['name']] = (data, labels, eeg_channels)

    if not cfg_results:
        return None

    meta = {
        "key": rec_id,
        "dataset_type": "cassette" if prefix == "SC" else "telemetry",
        "subject": f"{prefix}_{subject_id}",
        "night": int(night_id),
        "num_epochs": len(next(iter(cfg_results.values()))[1]),
    }

    return rec_id, meta, cfg_results


def build_lmdb(data_root, pre_root, configs, n_workers=8):
    psg_files = sorted(glob(os.path.join(data_root, "**", "*PSG.edf"), recursive=True))
    hyp_files = sorted(glob(os.path.join(data_root, "**", "*Hypnogram.edf"), recursive=True))
    assert len(psg_files) == len(hyp_files), "Mismatch between PSG and Hypnogram file counts."
    pairs = list(zip(psg_files, hyp_files))
    print(f"Found {len(pairs)} PSG-Hypnogram pairs.")

    # Open one LMDB env per config
    envs = {}
    for cfg in configs:
        lmdb_path = os.path.join(pre_root, f"{dataset_name}_{cfg['name']}")
        os.makedirs(lmdb_path, exist_ok=True)
        envs[cfg['name']] = lmdb.open(lmdb_path, map_size=1099511627776)

    all_meta = {cfg['name']: [] for cfg in configs}
    channels_saved = {cfg['name']: False for cfg in configs}

    args = [(psg, hyp, configs) for psg, hyp in pairs]

    print(f"Processing {len(pairs)} pairs in parallel (workers={n_workers})...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_process_single_pair, a) for a in args]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Pairs"):
            result = future.result()
            if result is None:
                continue

            rec_id, meta, cfg_results = result

            for cfg_name, (data, labels, eeg_channels) in cfg_results.items():
                with envs[cfg_name].begin(write=True) as txn:
                    txn.put(rec_id.encode('ascii'), pickle.dumps({
                        'data': data.astype(np.float32),
                        'labels': labels.astype(np.int64),
                    }))
                all_meta[cfg_name].append(meta)

                if not channels_saved[cfg_name]:
                    lmdb_path = os.path.join(pre_root, f"{dataset_name}_{cfg_name}")
                    json.dump(eeg_channels, open(os.path.join(lmdb_path, "channels.json"), "w"), indent=4)
                    channels_saved[cfg_name] = True

    for cfg in configs:
        envs[cfg['name']].close()

    dfs = []
    for cfg in configs:
        df = pd.DataFrame(all_meta[cfg['name']])
        csv_path = os.path.join(pre_root, f"{dataset_name}_{cfg['name']}_metadata.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved metadata: {csv_path}")
        dfs.append(df)

    return dfs


def generate_splits(df, split_path):
    print("\n--- Generating Splits ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def save_json(split_dict, filename):
        out_data = {}
        for phase, sub_df in split_dict.items():
            out_data[phase] = [
                {"key": row['key'], "length": int(row['num_epochs']), "subject": row['subject']}
                for _, row in sub_df.iterrows()
            ]
        with open(os.path.join(split_path, filename), 'w') as f:
            json.dump(out_data, f, indent=4)
        print(f"  Saved: {filename}")

    # --- Sleep Cassette (SC) - 5 Fold CV ---
    sc_df = df[df['dataset_type'] == 'cassette']
    sc_subjects = sc_df['subject'].unique()
    print(f"\nCassette 5-Fold CV (subjects={len(sc_subjects)})...")
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(sc_subjects)):
        test_subs = sc_subjects[test_idx]
        train_subs, val_subs = train_test_split(sc_subjects[train_idx], test_size=0.2, random_state=RANDOM_STATE)
        save_json({
            "train": sc_df[sc_df['subject'].isin(train_subs)],
            "val":   sc_df[sc_df['subject'].isin(val_subs)],
            "test":  sc_df[sc_df['subject'].isin(test_subs)],
        }, f"5class_cassette_fold{fold_idx}.json")

    # --- Sleep Telemetry (ST) - 5 Fold CV ---
    st_df = df[df['dataset_type'] == 'telemetry']
    st_subjects = st_df['subject'].unique()
    print(f"\nTelemetry 5-Fold CV (subjects={len(st_subjects)})...")
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(st_subjects)):
        test_subs = st_subjects[test_idx]
        train_subs, val_subs = train_test_split(st_subjects[train_idx], test_size=0.2, random_state=RANDOM_STATE)
        save_json({
            "train": st_df[st_df['subject'].isin(train_subs)],
            "val":   st_df[st_df['subject'].isin(val_subs)],
            "test":  st_df[st_df['subject'].isin(test_subs)],
        }, f"5class_telemetry_fold{fold_idx}.json")

    # --- Transfer: Train SC -> Test ST ---
    print("\nTransfer split (SC -> ST)...")
    train_subs, val_subs = train_test_split(sc_subjects, test_size=0.2, random_state=RANDOM_STATE)
    save_json({
        "train": sc_df[sc_df['subject'].isin(train_subs)],
        "val":   sc_df[sc_df['subject'].isin(val_subs)],
        "test":  st_df,
    }, "5class_transfer_SCtoST.json")

    # --- Within-Subject: Night 1 Train -> Night 2 Test (single fold, all valid subjects) ---
    print("\nWithin-subject split (Night 1 -> Night 2)...")
    sc_df = df[df['dataset_type'] == 'cassette']
    subject_nights = sc_df.groupby('subject')['night'].apply(set).to_dict()
    valid_subjects = sorted(sub for sub, nights in subject_nights.items() if 1 in nights and 2 in nights)

    train_df = sc_df[(sc_df['subject'].isin(valid_subjects)) & (sc_df['night'] == 1)]
    test_df  = sc_df[(sc_df['subject'].isin(valid_subjects)) & (sc_df['night'] == 2)]

    save_json({"train": train_df, "val": train_df, "test": test_df},
              "all_cassette_subjectid_fold0.json")
    save_json({"train": train_df, "val": train_df, "test": test_df},
              "all_cassette_sleep_fold0.json")

    print(f"  Within-subject fold: {len(valid_subjects)} subjects, "
          f"{len(train_df)} train epochs, {len(test_df)} test epochs.")
    
    # --- Within-Subject: Night 1 Train -> Night 2 Test (5 fold, 20 valid subjects per fold) ---
    np.random.seed(RANDOM_STATE)

    for i in range(5):
        # sample 20 subjects 
        fold_valid_subject = sorted(np.random.choice(valid_subjects, size=20, replace=False))
        train_fold_df = sc_df[(sc_df['subject'].isin(fold_valid_subject)) & (sc_df['night'] == 1)]
        test_fold_df  = sc_df[(sc_df['subject'].isin(fold_valid_subject)) & (sc_df['night'] == 2)]

        save_json({"train": train_fold_df, "val": train_fold_df, "test": test_fold_df},
              f"20subjects_cassette_subjectid_fold{i}.json")
        save_json({"train": train_fold_df, "val": train_fold_df, "test": test_fold_df},
                f"20subjects_cassette_sleep_fold{i}.json")

        print(fold_valid_subject)
        print(f"  Within-subject fold: {len(fold_valid_subject)} subjects, "
            f"{len(train_df)} train epochs, {len(test_df)} test epochs.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true", help="Run LMDB preprocessing")
    parser.add_argument("--splits", action="store_true", help="Generate dataset splits")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for preprocessing")
    args = parser.parse_args()

    if not args.preprocess and not args.splits:
        parser.print_help()
        raise SystemExit("\nSpecify --preprocess, --splits, or both.")

    data_root, pre_root, split_path = get_paths()

    if args.preprocess:
        download_dataset(data_root)
        processing_configs = [
            # {'name': 'raw_norm_100', 'fs': 100, 'norm': True},
            {'name': 'raw_norm_200', 'fs': 200, 'norm': True},
            # {'name': 'csbrain_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'csbrain_200'},
            # {'name': 'labram_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'labram_200'},
            # {'name': 'cbramod_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'cbramod_200'}
        ]
        dfs = build_lmdb(data_root, pre_root, processing_configs, n_workers=args.workers)

    if args.splits:
        processing_configs = [
            {'name': 'raw_norm_200', 'fs': 200, 'norm': True},
        ]
        csv_path = os.path.join(pre_root, f"{dataset_name}_{processing_configs[0]['name']}_metadata.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata CSV not found at {csv_path}. Run --preprocess first.")
        df = pd.read_csv(csv_path)
        generate_splits(df, split_path)
