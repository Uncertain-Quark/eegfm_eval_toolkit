# Dataset for https://figshare.com/articles/dataset/EEG_Data_New/4244171 dataset
# Consists of resting state and task (P 300) data
# https://scholar.google.com/scholar?cluster=8556220843927428512&hl=en&oi=scholarr

# The dataset will have two modes
# 1. Load the Eyes closed/Eyes open resting state data
# 2. Use the task based EEG data

import torch
import lmdb
import pickle
import numpy as np
import random
from torch.utils.data import Dataset

class mdd_mal(Dataset):
    def __init__(self, lmdb_path, data_list, seq_len_sec=5, fs=256, percent_train_per_subject: float=None, 
                 n_samples_per_subject=None, seed: int=42, channel_idx: list=None, model_type: str=None):
        """
        Args:
            lmdb_path (str): Path to the folder containing LMDB files.
            data_list (list): List of dicts (from your JSON split) containing metadata.
            seq_len_sec (float): Length of the window in seconds.
            fs (int): Sampling frequency.
        """
        super(mdd_mal, self).__init__()
        
        self.lmdb_path = lmdb_path
        self.seq_len = int(seq_len_sec * fs)
        self.env = None # Placeholder for LMDB environment
        self.seed = seed
        self.channel_idx = channel_idx
        self.model_type = model_type
        rng = random.Random(seed)

        # --- Pre-calculate all valid windows (Virtual Indexing) ---
        # We flatten the dataset: instead of list of Files, we make a list of Windows.
        # index 0 -> File A, window 0
        # index 1 -> File A, window 1
        # index N -> File B, window 0
        self.windows = []

        for item in data_list:
            total_samples = item['length']
            key = item['key']
            label = item['label']

            # Logic: Non-overlapping windows (stride = seq_len)
            # You can change step=self.seq_len//2 for 50% overlap if data is scarce
            starts = list(range(0, total_samples - self.seq_len + 1, self.seq_len))
            rng.shuffle(starts)
            
            if (percent_train_per_subject is not None and percent_train_per_subject<1.0) or n_samples_per_subject is not None:
                n_keep = min(int(percent_train_per_subject*len(starts)) if percent_train_per_subject is not None else n_samples_per_subject, len(starts))
                starts = starts[:n_keep]
            
            for start in starts:
                self.windows.append({
                    "key": key,
                    "start": start,
                    "label": label
                })
                
        print(f"Dataset loaded: {len(data_list)} files transformed into {len(self.windows)} windows.")

    def _init_db(self):
        """
        Lazy initialization of LMDB environment. 
        This is necessary because LMDB environments cannot be pickled 
        and passed to DataLoader workers. Each worker must open its own connection.
        """
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path, 
                readonly=True, 
                lock=False, 
                readahead=False, 
                meminit=False
            )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # 1. Ensure DB is open (once per worker)
        self._init_db()

        # 2. Get window metadata
        window_meta = self.windows[idx]
        key = window_meta['key'].encode('ascii')
        start = window_meta['start']
        label = window_meta['label']

        # 3. Read from LMDB
        with self.env.begin(write=False) as txn:
            raw_bytes = txn.get(key)
            if raw_bytes is None:
                raise ValueError(f"Key {key} not found in LMDB")
            
            data_dict = pickle.loads(raw_bytes)
            full_data = data_dict['data'] # Could be a dict or a numpy array

            # --- Extract standard arrays whether it's the new or old format ---
            if type(full_data) == dict:
                raw_eeg = full_data["data"]
                channel_idx_tensor = torch.tensor(full_data["channel_idx"], dtype=torch.long).clone()
            else:
                raw_eeg = full_data
                channel_idx_tensor = None # Old format doesn't have this

            # 4. Slice the specific window (APPLIES TO BOTH FORMATS NOW)
            if self.channel_idx is None:
                sliced_eeg = raw_eeg[:, start : start + self.seq_len]
            else:
                sliced_eeg = raw_eeg[self.channel_idx, start : start + self.seq_len]

            # 5. Fix memory lock and convert to Tensor
            x = torch.from_numpy(sliced_eeg.copy()).float()
            y = torch.tensor(label, dtype=torch.long)

            # 6. Return as a Dictionary
            batch_dict = {
                "inputs": x,
                "labels": y
            }
            
            # Only add channel_ids if they actually exist in this dataset
            if channel_idx_tensor is not None:
                batch_dict["channel_ids"] = channel_idx_tensor

            return batch_dict