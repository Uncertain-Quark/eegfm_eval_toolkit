# sleep EDFX dataset 
# https://www.physionet.org/content/sleep-edfx/1.0.0/
# consists of 2 EEG electrodes 
# Different tasks that can be defined
# ---- Between subjects sleep staging
# ---- Within subject between days sleep staging (could look into personalization aspect)
# Based on demographic details, we could attempt to verify whether the demographic of trained subjects impacts the downstream performance of the models

import torch
import random
import math
import lmdb
import pickle
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info
from sklearn.model_selection import train_test_split
from collections import defaultdict
import copy 

class sleep_edf(IterableDataset):
    def __init__(self, lmdb_path, data_list, shuffle=True, buffer_size=500, percent_train_per_subject=None, 
                 n_samples_per_subject=None, seed=42, channel_idx: list=None, model_type: str=None,
                 task_type: str='sleep'):
        """
        Args:
            lmdb_path (str): Path to the LMDB folder.
            data_list (list): List of dicts from the JSON split (contains 'key', 'length', 'subject').
            shuffle (bool): Whether to shuffle files and the buffer.
            buffer_size (int): Size of the shuffle buffer.
        """
        super(sleep_edf, self).__init__()

        self.lmdb_path = lmdb_path
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.percent_train_per_subject = percent_train_per_subject
        self.n_samples_per_subject = n_samples_per_subject
        self.seed = seed
        self.channel_idx = channel_idx
        self.model_type = model_type

        self.task_type = task_type
        # self.subject_id_to_labels = subject_id_to_labels

        # Create a deep copy to avoid modifying the input list in place
        working_data_list = copy.deepcopy(data_list)
        
        # -----------------------------------------------------------
        # LOGIC FIX: Distribute n_samples_per_subject across files
        # -----------------------------------------------------------
        self.data_list = []
        
        if self.task_type == 'subject_id':
            # Extract unique subjects and sort them for deterministic mapping
            unique_subjects = list(set([d['subject'] for d in working_data_list]))
            unique_subjects.sort()
            
            # Map each subject to an integer from 0 to N-1
            self.subject_id_to_labels = {sub: idx for idx, sub in enumerate(unique_subjects)}
            # Optional: print mapping for debugging
            # print(f"[{self.task_type.upper()}] Computed Subject Mapping: {self.subject_id_to_labels}")
        else:
            self.subject_id_to_labels = None

        # Group files by subject
        dict_subject_list = defaultdict(list)
        for d in working_data_list:
            dict_subject_list[d["subject"]].append(d)

        # if n_samples_per_subject is not None:
        #     for subject, files in dict_subject_list.items():
        #         # Sort files (e.g., by key/night) to ensure deterministic selection
        #         files.sort(key=lambda x: x['key'])
                
        #         samples_needed = n_samples_per_subject
                
        #         for f in files:
        #             if samples_needed <= 0:
        #                 continue # We have enough data for this subject, skip remaining files
                    
        #             # We assign a 'quota' to this file. 
        #             # This tells __iter__ "You are allowed to take up to X samples from this file".
        #             # If samples_needed is huge, the quota is huge, but __iter__ will be capped by actual file length.
        #             f['sample_quota'] = samples_needed
        #             self.data_list.append(f)
                    
        #             # Deduct the *maximum possible* samples this file could contribute (its length)
        #             # from the counter so the next file knows how much is left to provide.
        #             samples_needed -= f['length']
        if n_samples_per_subject is not None:
            for subject, files in dict_subject_list.items():
                # Sort files to ensure deterministic behavior
                files.sort(key=lambda x: x['key'])
                
                # 1. Get total duration of all nights for this subject
                total_subject_len = sum(f['length'] for f in files)
                
                # 2. Distribute quota proportionally
                allocated_so_far = 0
                for i, f in enumerate(files):
                    # Calculate ratio: (File Length / Total Subject Length)
                    ratio = f['length'] / total_subject_len
                    
                    if i == len(files) - 1:
                        # Last file gets the remainder (handles rounding errors)
                        quota = n_samples_per_subject - allocated_so_far
                    else:
                        # Other files get their fair share
                        quota = int(round(ratio * n_samples_per_subject))
                    
                    # Store the quota
                    if quota > 0:
                        f['sample_quota'] = quota
                        self.data_list.append(f)
                        allocated_so_far += quota
        else:
            # If no fixed sample count, keep all files as is
            self.data_list = working_data_list

        # -----------------------------------------------------------
        # Calculate Total Samples for __len__
        # -----------------------------------------------------------
        self.total_samples = 0
        mult_factor = 1 if percent_train_per_subject is None else percent_train_per_subject
        
        for item in self.data_list:
            # If we have a specific quota, use it (capped by file length to be realistic)
            if 'sample_quota' in item:
                expected_yield = min(item['sample_quota'], item['length'])
                self.total_samples += expected_yield
            else:
                self.total_samples += int(item['length'] * mult_factor)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        rng = random.Random(self.seed)
        
        # 1. SETUP
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        
        # 2. WORKER SPLIT
        worker_info = get_worker_info()
        if worker_info is None:  
            my_files = self.data_list
        else:  
            per_worker = int(math.ceil(len(self.data_list) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.data_list))
            my_files = self.data_list[iter_start:iter_end]

        # 3. TIER 1 SHUFFLE (Files)
        my_files = list(my_files)
        if self.shuffle:
            rng.shuffle(my_files)

        # 4. STREAMING GENERATOR
        def stream_epochs():
            with env.begin() as txn:
                for item in my_files:
                    key = item['key'].encode('ascii')
                    raw_bytes = txn.get(key)
                    if raw_bytes is None: continue
                    
                    entry = pickle.loads(raw_bytes)
                    data_all = entry['data']
                    labels_all = entry['labels']

                    # Filter out -1 labels
                    indices_of_interest = np.where(labels_all != -1)[0]
                    data_all = data_all[indices_of_interest]
                    labels_all = labels_all[indices_of_interest]
                    
                    n_total = len(labels_all)
                    if n_total < 2: continue # Skip empty/single-sample files

                    # ------------------------------------------------
                    # LOGIC FIX: Determine Target Size based on Quota
                    # ------------------------------------------------
                    quota = item.get('sample_quota', None)
                    
                    if quota is not None:
                        # Case A: Fixed N samples (derived from init)
                        target_size = min(quota, n_total)
                    elif self.percent_train_per_subject is not None:
                        # Case B: Percentage
                        target_size = int(n_total * self.percent_train_per_subject)
                    else:
                        # Case C: All data
                        target_size = n_total

                    # Perform Split if we need to subsample
                    if target_size < n_total:
                        try:
                            keep_idxs, _ = train_test_split(
                                np.arange(n_total),
                                train_size=target_size,
                                stratify=labels_all,
                                random_state=self.seed 
                            )
                        except ValueError:
                            # Fallback if stratification fails (e.g. rare class)
                            keep_idxs, _ = train_test_split(
                                np.arange(n_total),
                                train_size=target_size,
                                random_state=self.seed
                            )
                        
                        data_all = data_all[keep_idxs]
                        labels_all = labels_all[keep_idxs]

                    # ------------------------------------------------
                    # Yielding
                    # ------------------------------------------------
                    n_epochs = data_all.shape[0]
                    indices = list(range(n_epochs))
                    
                    if self.shuffle:
                        rng.shuffle(indices)
                        
                    for i in indices:
                        x = data_all[i].copy() 
                        # y = labels_all[i]
                        if self.task_type == 'subject_id':
                            if self.subject_id_to_labels is None:
                                raise ValueError("subject_id_to_labels dict must be provided for the subject_id task.")
                            y = self.subject_id_to_labels[item['subject']]
                        else:
                            y = labels_all[i]

                        if self.channel_idx is not None: 
                            if "cbramod" in self.model_type:
                                temp_arr = np.zeros_like(x)
                                temp_arr[self.channel_idx] = x[self.channel_idx]
                                x = temp_arr
                            else:
                                x = x[self.channel_idx]
                        yield x, y

        # 5. TIER 2 SHUFFLE (Buffer)
        iterator = stream_epochs()
        
        if self.shuffle:
            buffer = []
            try:
                for _ in range(self.buffer_size):
                    buffer.append(next(iterator))
            except StopIteration:
                pass 
            
            for item in iterator:
                idx = rng.randint(0, len(buffer) - 1)
                yield buffer[idx]
                buffer[idx] = item
            
            rng.shuffle(buffer)
            for item in buffer:
                yield item
        else:
            yield from iterator