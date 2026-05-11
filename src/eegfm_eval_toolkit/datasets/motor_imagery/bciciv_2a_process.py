import os
import json
import mne
import numpy as np
import tqdm
import scipy.io
from scipy import signal
import pandas as pd
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed

from biodl.utils.make_ssl_data import process_ssl_data

# --- Configuration ---
dataset_name = "bciciv_2a"
RANDOM_STATE = 42
SUBJECTS = range(1, 10)

EVENT_ID_MAP = {'769': 0, '770': 1, '771': 2, '772': 3}
EVENT_START_SEC = 1.0
EVENT_END_SEC = 4.0

EEG_CHANNEL_MAP = {
    'EEG-Fz': "Fz",   'EEG-0': "Fc3",  'EEG-1': "Fc1",  'EEG-2': "Fcz",
    'EEG-3': "Fc2",   'EEG-4': "Fc4",  'EEG-5': "C5",   'EEG-C3': "C3",
    'EEG-6': "C1",    'EEG-Cz': "Cz",  'EEG-7': "C2",   'EEG-C4': "C4",
    'EEG-8': "C6",    'EEG-9': "CP3",  'EEG-10': "CP1", 'EEG-11': "CPz",
    'EEG-12': "CP2",  'EEG-13': "CP4", 'EEG-14': "P1",  'EEG-Pz': "Pz",
    'EEG-15': "P2",   'EEG-16': "POz",
}


def get_paths():
    root = os.getenv("BIODL_RAW_DATA_ROOT", "./data")
    data_root = os.path.join(root, dataset_name)
    pre_root = os.getenv("BIODL_PREPROCESSED_DATA_ROOT", "./data_preprocessed")
    split_path = os.path.join(os.getenv("BIODL_SPLIT_PATH", "./splits"), dataset_name)
    os.makedirs(split_path, exist_ok=True)
    return data_root, pre_root, split_path


def download_dataset(data_root):
    extraction_dir = os.path.join(data_root, "extracted")
    labels_dir = os.path.join(data_root, "labels")

    if os.path.exists(extraction_dir) and os.path.exists(labels_dir):
        print(f"Dataset found at {data_root}, skipping download.")
        return

    print(f"Downloading dataset to {data_root}...")
    os.makedirs(extraction_dir, exist_ok=True)

    data_url = "https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip"
    labels_url = "https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip"

    if os.system(f"wget -c -P {data_root} {data_url}") != 0:
        raise RuntimeError("Failed to download data zip.")
    os.system(f"unzip -o {os.path.join(data_root, 'BCICIV_2a_gdf.zip')} -d {extraction_dir}")

    os.makedirs(labels_dir, exist_ok=True)
    if os.system(f"wget -c -P {labels_dir} {labels_url}") != 0:
        raise RuntimeError("Failed to download labels zip.")
    os.system(f"unzip -o {os.path.join(labels_dir, 'true_labels.zip')} -d {extraction_dir}")

    print("Download and extraction complete.")


def compute_spectrogram(data, fs):
    _, _, Zxx = signal.stft(data, fs=fs, nperseg=fs // 2)
    return np.abs(Zxx).astype(np.float32)


def _load_and_epoch_subject(args):
    """
    Loads, filters, and epochs all sessions for a single subject.
    Returns a dict keyed by config name with lists of (key, X, y).
    """
    subject_id, raw_dir, configs = args

    results = {cfg['name']: {'X': [], 'y': [], 'keys': [], 'channels': None} for cfg in configs}
    subject_keys = []

    for session_type, fname in [('T', f"A0{subject_id}T.gdf"), ('E', f"A0{subject_id}E.gdf")]:
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            print(f"Missing {fpath}")
            continue

        try:
            raw = mne.io.read_raw_gdf(fpath, preload=True, verbose=False)
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue

        non_eog = mne.pick_channels(raw.ch_names, include=[ch for ch in raw.ch_names if "eog" not in ch.lower()])
        raw.pick(non_eog)
        if raw.info['bads']:
            raw.interpolate_bads()

        ch_order = [EEG_CHANNEL_MAP[ch] for ch in raw.info["ch_names"]]

        raw.set_eeg_reference(ref_channels='average')
        raw.notch_filter(50.0, method='iir', phase='zero', verbose=False)
        raw.filter(0.3, 50.0, method='fir', phase='zero', verbose=False)

        if session_type == 'T':
            all_events, all_event_id = mne.events_from_annotations(raw, verbose=False)
            reject_id = all_event_id.get('1023', None)
            class_ids = [all_event_id[k] for k in ['769', '770', '771', '772'] if k in all_event_id]
            id_to_label = {v: EVENT_ID_MAP[k] for k, v in all_event_id.items() if k in EVENT_ID_MAP}
            artifact_samples = all_events[all_events[:, 2] == reject_id, 0] if reject_id is not None else np.array([])

            fs_raw = int(raw.info['sfreq'])
            trial_len_samples = int(4.0 * fs_raw)
            valid_trials = []

            for event in all_events:
                eid, start_samp = event[2], event[0]
                if eid not in class_ids:
                    continue
                end_samp = start_samp + trial_len_samples
                if len(artifact_samples) > 0 and np.any((artifact_samples >= start_samp) & (artifact_samples < end_samp)):
                    continue
                valid_trials.append((start_samp, id_to_label[eid]))

        else:
            def find_event_code(map_dict, codes):
                for code in codes:
                    if str(code) in map_dict:
                        return map_dict[str(code)]
                return None

            mat_file = fpath.replace(".gdf", ".mat")
            if not os.path.exists(mat_file):
                print(f"CRITICAL: Labels file {mat_file} not found. Skipping.")
                continue

            mat_data = scipy.io.loadmat(mat_file)
            true_labels = mat_data['classlabel'].flatten()

            events, event_id_map = mne.events_from_annotations(raw, verbose=False)
            cue_code = find_event_code(event_id_map, ['783'])
            if cue_code is None:
                print(f"Error: Could not find Cue event (783) in {fname}")
                continue

            cue_events = events[events[:, 2] == cue_code]
            n_trials = min(len(cue_events), len(true_labels))
            if n_trials != 288:
                print(f"Warning: Expected 288 trials, found {n_trials} in {fname}")

            valid_trials = [
                (cue_events[i, 0], int(true_labels[i] - 1))
                for i in range(n_trials)
                if true_labels[i] in [1, 2, 3, 4]
            ]

        # Build resampled cache per unique fs
        resampled_cache = {}
        for fs in set(c['fs'] for c in configs):
            if int(raw.info["sfreq"]) != fs:
                r_tmp = raw.copy().resample(fs, method="polyphase", verbose=False, n_jobs=4)
                resampled_cache[fs] = (r_tmp, fs / raw.info["sfreq"])
            else:
                resampled_cache[fs] = (raw, 1.0)

        for i, (orig_sample, label) in enumerate(valid_trials):
            key_str = f"S{subject_id:03d}_{session_type}_{label}_{i:03d}"
            subject_keys.append(key_str)

            for cfg in configs:
                fs = cfg['fs']
                raw_res, scale = resampled_cache[fs]

                event_start_sec = cfg.get("EVENT_START_SEC", EVENT_START_SEC)
                event_end_sec = cfg.get("EVENT_END_SEC", EVENT_END_SEC)

                trigger_sample = int(orig_sample * scale)
                start_idx = trigger_sample + int(event_start_sec * fs)
                end_idx = trigger_sample + int(event_end_sec * fs)

                if end_idx > raw_res.n_times:
                    continue

                data_chunk = raw_res.get_data(units="uV")[:, start_idx:end_idx]
                expected_len = int((event_end_sec - event_start_sec) * fs)

                if data_chunk.shape[1] != expected_len:
                    if abs(data_chunk.shape[1] - expected_len) < 5 and data_chunk.shape[1] > expected_len:
                        data_chunk = data_chunk[:, :expected_len]
                    else:
                        continue

                if cfg.get('ssl') is not None:
                    data_chunk, out_channels = process_ssl_data(data_chunk, ch_order, ssl=cfg['ssl'])
                    if results[cfg['name']]['channels'] is None:
                        results[cfg['name']]['channels'] = out_channels

                if cfg['norm']:
                    mu = np.mean(data_chunk, axis=1, keepdims=True)
                    std = np.std(data_chunk, axis=1, keepdims=True)
                    std[std == 0] = 1.0
                    data_chunk = (data_chunk - mu) / std

                final_data = compute_spectrogram(data_chunk, fs) if cfg.get('spec', False) else data_chunk.astype(np.float32)

                results[cfg['name']]['X'].append(final_data)
                results[cfg['name']]['y'].append(label)
                results[cfg['name']]['keys'].append(key_str)

                if results[cfg['name']]['channels'] is None:
                    results[cfg['name']]['channels'] = ch_order

    return subject_id, subject_keys, results


def process_and_store_memory(data_root, pre_root, configs, n_workers=4):
    raw_dir = os.path.join(data_root, "extracted")

    merged = {cfg['name']: {'X': [], 'y': [], 'keys': [], 'channels': None} for cfg in configs}
    subject_to_keys = {}

    args = [(sid, raw_dir, configs) for sid in SUBJECTS]

    print(f"Processing {len(SUBJECTS)} subjects in parallel (workers={n_workers})...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_load_and_epoch_subject, a): a[0] for a in args}
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Subjects"):
            subject_id, subject_keys, results = future.result()
            subject_to_keys[subject_id] = subject_keys
            for cfg_name, data in results.items():
                merged[cfg_name]['X'].extend(data['X'])
                merged[cfg_name]['y'].extend(data['y'])
                merged[cfg_name]['keys'].extend(data['keys'])
                if merged[cfg_name]['channels'] is None and data['channels'] is not None:
                    merged[cfg_name]['channels'] = data['channels']

    print("Saving preprocessed arrays to disk...")
    for cfg in configs:
        cfg_name = cfg['name']
        data = merged[cfg_name]

        X_all = np.stack(data['X'])
        y_all = np.array(data['y'])
        keys_all = np.array(data['keys'])

        save_dir = os.path.join(pre_root, f"{dataset_name}_{cfg_name}")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, "data.npz")
        print(f"  {cfg_name} -> {save_path} | Shape: {X_all.shape}")
        np.savez(save_path, X=X_all, y=y_all, keys=keys_all)

        ch_path = os.path.join(save_dir, "channels.json")
        json.dump(data['channels'], open(ch_path, "w"), indent=4)

    keys_path = os.path.join(pre_root, f"{dataset_name}_subject_keys.json")
    json.dump({str(k): v for k, v in subject_to_keys.items()}, open(keys_path, "w"), indent=4)
    print(f"Saved subject key map to {keys_path}")

    return subject_to_keys


def load_subject_keys(pre_root):
    keys_path = os.path.join(pre_root, f"{dataset_name}_subject_keys.json")
    if not os.path.exists(keys_path):
        raise FileNotFoundError(
            f"Subject key map not found at {keys_path}. Run preprocessing first."
        )
    raw = json.load(open(keys_path))
    return {int(k): v for k, v in raw.items()}


def parse_keys_to_df(subject_to_keys):
    data = []
    for sub, keys in subject_to_keys.items():
        for k in keys:
            parts = k.split('_')
            data.append({
                "key": k,
                "subject": int(parts[0][1:]),
                "session": parts[1],
                "label": parts[2],
            })
    return pd.DataFrame(data)


def generate_all_splits(subject_to_keys, split_path):
    print("Parsing metadata from keys...")
    df = parse_keys_to_df(subject_to_keys)
    subjects = sorted(df['subject'].unique())

    def save_split(name, train_df, val_df, test_df):
        out = {
            "train": train_df['key'].tolist(),
            "val": val_df['key'].tolist(),
            "test": test_df['key'].tolist(),
        }
        path = os.path.join(split_path, f"{name}.json")
        with open(path, 'w') as f:
            json.dump(out, f, indent=4)
        print(f"  Saved: {name} | Tr:{len(train_df)} V:{len(val_df)} Ts:{len(test_df)}")

    # --- 4-class within-subject (random 4 subjects, 5 folds) ---
    np.random.seed(RANDOM_STATE)
    for fold in range(5):
        subjects_4 = sorted(np.random.choice(subjects, 4, replace=False))
        ws_train = df[(df['session'] == 'T') & (df['subject'].isin(subjects_4))]
        ws_test = df[(df['session'] == 'E') & (df['subject'].isin(subjects_4))]
        ws_tr, ws_val = train_test_split(ws_train, test_size=0.2, random_state=RANDOM_STATE, stratify=ws_train['subject'])
        save_split(f"4class_within_subject_id_fold{fold}", ws_tr, ws_val, ws_test)
        save_split(f"4class_within_subject_motor_fold{fold}", ws_tr, ws_val, ws_test)

    # --- 4-class within-subject (all subjects, T->E) ---
    print("\n--- Within-Subject (T vs E) ---")
    ws_train = df[df['session'] == 'T']
    ws_test = df[df['session'] == 'E']
    ws_tr, ws_val = train_test_split(ws_train, test_size=0.2, random_state=RANDOM_STATE, stratify=ws_train['subject'])
    save_split("4class_within_subject_fold0", ws_tr, ws_val, ws_test)

    # --- Per-subject splits (T->E) ---
    print("\n--- Per-Subject Independent Splits ---")
    for i, sub in enumerate(subjects):
        sub_df = df[df['subject'] == sub]
        sub_train = sub_df[sub_df['session'] == 'T']
        sub_test = sub_df[sub_df['session'] == 'E']
        if len(sub_train) > 1:
            sub_tr, sub_val = train_test_split(sub_train, test_size=0.2, random_state=RANDOM_STATE, stratify=sub_train['label'])
        else:
            sub_tr, sub_val = sub_train, sub_train.iloc[:0]
        save_split(f"4class_subject_specific_fold{i}", sub_tr, sub_val, sub_test)

    # --- Per-subject splits (random T/E split) ---
    for i, sub in enumerate(subjects):
        sub_df = df[df['subject'] == sub]
        sub_train, sub_test = train_test_split(sub_df, test_size=0.2, random_state=RANDOM_STATE, stratify=sub_df['label'])
        sub_tr, sub_val = train_test_split(sub_train, test_size=0.2, random_state=RANDOM_STATE, stratify=sub_train['label'])
        save_split(f"4class_subject_specific_te_fold{i}", sub_tr, sub_val, sub_test)

    # --- Cross-subject LOSO ---
    print("\n--- Cross-Subject (LOSO) ---")
    for i, test_sub in enumerate(subjects):
        test_df = df[df['subject'] == test_sub]
        train_pool = df[df['subject'] != test_sub]
        tr_df, val_df = train_test_split(train_pool, test_size=0.2, random_state=RANDOM_STATE, stratify=train_pool['subject'])
        save_split(f"4class_cross_subject_fold{i}", tr_df, val_df, test_df)

    # --- Cross-subject session transfer (T->E) ---
    print("\n--- Cross-Subject Session Transfer (Train T -> Test E) ---")
    for i, test_sub in enumerate(subjects):
        test_df = df[(df['subject'] == test_sub) & (df['session'] == 'E')]
        train_pool = df[(df['subject'] != test_sub) & (df['session'] == 'T')]
        if len(train_pool) > 5:
            tr_df, val_df = train_test_split(train_pool, test_size=0.2, random_state=RANDOM_STATE, stratify=train_pool['subject'])
        else:
            tr_df, val_df = train_pool, train_pool.iloc[:0]
        save_split(f"4class_cross_subject_sess_transfer_fold{i}", tr_df, val_df, test_df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing")
    parser.add_argument("--splits", action="store_true", help="Generate dataset splits")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for preprocessing")
    args = parser.parse_args()

    if not args.preprocess and not args.splits:
        parser.print_help()
        raise SystemExit("\nSpecify --preprocess, --splits, or both.")

    data_root, pre_root, split_path = get_paths()

    if args.preprocess:
        download_dataset(data_root)
        processing_configs = [
            # {'name': 'raw_norm_3s_250', 'fs': 250, 'norm': True, 'spec': False},
            # {'name': 'raw_norm_3s_160', 'fs': 160, 'norm': True, 'spec': False},
            # {'name': 'raw_norm_4s_250', 'fs': 250, 'norm': True, 'spec': False, "EVENT_START_SEC": 0.0},
            {'name': 'raw_norm_4s_200', 'fs': 200, 'norm': True, 'spec': False, "EVENT_START_SEC": 0.0},
            # {'name': 'raw_norm_4s_160', 'fs': 160, 'norm': True, 'spec': False, "EVENT_START_SEC": 0.0}
            # {'name': 'spectrogram_250', 'fs': 250, 'norm': True, 'spec': True},
            # Downsampled version for lighter models
            # {'name': 'raw_norm_128', 'fs': 128, 'norm': True, 'spec': False}, 
            # {'name': 'cbramod_3s_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'cbramod_200'},
            # {'name': 'labram_3s_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'labram_200'},
            # {'name': 'csbrain_3s_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'csbrain_200'},
            # {'name': 'cbramod_4s_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'cbramod_200', "EVENT_START_SEC": 0.0},
            # {'name': 'labram_4s_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'labram_200', "EVENT_START_SEC": 0.0},
            # {'name': 'csbrain_4s_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'csbrain_200', "EVENT_START_SEC": 0.0}
        ]
        process_and_store_memory(data_root, pre_root, processing_configs, n_workers=args.workers)

    if args.splits:
        subject_to_keys = load_subject_keys(pre_root)
        generate_all_splits(subject_to_keys, split_path)
