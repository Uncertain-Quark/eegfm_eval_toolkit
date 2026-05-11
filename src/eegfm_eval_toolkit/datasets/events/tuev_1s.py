# # TUEV dataset

# import lmdb
# import pickle
# import numpy as np 
# import random 
# import hashlib
# import copy
# from collections import defaultdict
# from sklearn.model_selection import train_test_split
# from torch.utils.data import IterableDataset, get_worker_info

# class tuev_1s(IterableDataset):
#     def __init__(
#             self,
#             lmdb_path: str=None,
#             data_list: list=None,
#             shuffle: bool=True,
#             buffer_size: int=500,
#             seq_len: int=5,
#             fs: int=200,
#             percent_train_per_subject: float=None,
#             n_samples_per_subject: int=None,
#             seed: int=42,
#             channel_idx: list=None,
#             label_noise: float=None,
#             num_classes: int=6,
#             model_type: str=None,
#             labels_str: str="label_6way",
#             **kwargs
#     ):
        
#         super(tuev_1s, self).__init__()
        
#         self.lmdb_path = lmdb_path
#         self.shuffle = shuffle
#         self.buffer_size = buffer_size
#         self.seq_len = seq_len
#         self.fs = fs 
#         self.sample_length = int(fs * seq_len)
#         self.percent_train_per_subject = percent_train_per_subject
#         self.n_samples_per_subject = n_samples_per_subject
#         self.seed = seed
#         self.channel_idx = channel_idx
#         self.label_noise = label_noise
#         self.num_classes = num_classes
#         self.model_type = model_type
#         self.labels_str = labels_str

#         # -----------------------------------------------------------
#         # 1. Group by Subject & Distribute Quota
#         # -----------------------------------------------------------
#         working_data_list = copy.deepcopy(data_list)
#         self.data_list = []
        
#         # Group: Subject ID is the first part of the key (e.g., "aaaaa_s001..." -> "aaaaa")
#         dict_subject_list = defaultdict(list)
#         for d in working_data_list:
#             subject = d["key"].split("_")[0]
#             dict_subject_list[subject].append(d)

#         if n_samples_per_subject is not None:
#             for subject, files in dict_subject_list.items():
#                 # Sort to ensure deterministic allocation across runs
#                 files.sort(key=lambda x: x['key'])
                
#                 # This is the "Total Events" we want for this patient
#                 events_needed = n_samples_per_subject
                
#                 for f in files:
#                     if events_needed <= 0:
#                         continue
                    
#                     # 'length' in data_list corresponds to number of events in that file
#                     available_events = f['length']
                    
#                     # Calculate quota for this specific file
#                     quota = min(events_needed, available_events)
#                     f['event_quota'] = quota
                    
#                     self.data_list.append(f)
#                     events_needed -= quota
#         else:
#             self.data_list = working_data_list

#         # -----------------------------------------------------------
#         # 2. Calculate Total Length
#         # -----------------------------------------------------------
#         self.length = 0
#         mult_factor = 1 if percent_train_per_subject is None else percent_train_per_subject

#         for d in self.data_list:
#             if 'event_quota' in d:
#                 # If we have a hard quota, that is the exact contribution
#                 self.length += d['event_quota']
#             else:
#                 # Otherwise use percentage logic
#                 self.length += int(d["length"] * mult_factor)

#     def _get_worker_slice(self):
#         worker_info = get_worker_info()
#         if worker_info is None:
#             return self.data_list
        
#         per_worker = int(np.ceil(len(self.data_list) / float(worker_info.num_workers)))
#         worker_id = worker_info.id
#         start = worker_id * per_worker
#         end = min(start + per_worker, len(self.data_list))
#         return self.data_list[start:end]
    
#     def __len__(self):
#         return self.length
    
#     def __iter__(self):
#         rng = random.Random(self.seed)
#         worker_data_list = self._get_worker_slice()
        
#         # Copy list to avoid shared memory issues
#         worker_data_list = list(worker_data_list)
#         if self.shuffle:
#             rng.shuffle(worker_data_list)

#         env = lmdb.open(
#             self.lmdb_path, 
#             readonly=True, 
#             lock=False, 
#             readahead=False, 
#             meminit=False
#         )
        
#         def stream_chunks():
#             with env.begin() as txn:
#                 for item in worker_data_list:
#                     key = item['key'].encode('ascii')
#                     raw_bytes = txn.get(key)
#                     if raw_bytes is None: continue

#                     file_obj = pickle.loads(raw_bytes)
#                     full_data = file_obj["data"]     # (Channels, Time)
#                     segments_meta = file_obj["metadata"] # List of dicts
#                     full_len = full_data.shape[1]
                    
#                     n_total = len(segments_meta)
#                     if n_total == 0: continue

#                     # ------------------------------------------------
#                     # DETERMINE SUBSET OF EVENTS TO PROCESS
#                     # ------------------------------------------------
#                     quota = item.get('event_quota', None)
                    
#                     if quota is not None:
#                         # Case A: Fixed quota from n_samples_per_subject
#                         target_size = quota
#                     elif self.percent_train_per_subject is not None:
#                         # Case B: Percentage
#                         target_size = int(n_total * self.percent_train_per_subject)
#                     else:
#                         # Case C: Use all
#                         target_size = n_total

#                     # Perform Stratified Split if we are subsampling
#                     if target_size < n_total and target_size > 0:
#                         labels = [s[self.labels_str] for s in segments_meta]
#                         try:
#                             keep_idxs, _ = train_test_split(
#                                 np.arange(n_total),
#                                 train_size=target_size,
#                                 stratify=labels,
#                                 random_state=self.seed 
#                             )
#                         except ValueError:
#                             # Fallback for rare classes
#                             keep_idxs, _ = train_test_split(
#                                 np.arange(n_total),
#                                 train_size=target_size,
#                                 random_state=self.seed
#                             )
                        
#                         # Sort indices to keep processing order consistent (optimizes cache slightly)
#                         keep_idxs = sorted(keep_idxs)
#                         segments_meta = [segments_meta[i] for i in keep_idxs]
                    
#                     # ------------------------------------------------
#                     # NOISE INJECTION
#                     # ------------------------------------------------
#                     if self.label_noise is not None and self.label_noise > 0.0:
#                         seed_str = f"{item['key']}_{self.seed}"
#                         file_seed = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest(), 16) % (2**32)
#                         file_rng = np.random.RandomState(file_seed)

#                         for seg in segments_meta:
#                             if file_rng.rand() < self.label_noise:
#                                 original = int(seg[self.labels_str])
#                                 # Candidates: 1..num_classes excluding original
#                                 candidates = [c for c in range(1, self.num_classes + 1) if c != original]
#                                 if candidates:
#                                     seg[self.labels_str] = file_rng.choice(candidates)

#                     # ------------------------------------------------
#                     # DATA EXTRACTION
#                     # ------------------------------------------------
#                     # Optional: Shuffle events within the file?
#                     if self.shuffle:
#                         rng.shuffle(segments_meta)

#                     for seg in segments_meta:
#                         # 1. Geometry
#                         ann_start = int(seg["start"] * self.fs)
#                         ann_end = int(seg["end"] * self.fs)
#                         center = (ann_start + ann_end) // 2
                        
#                         half_len = self.sample_length // 2
#                         win_start = center - half_len
#                         win_end = win_start + self.sample_length
                        
#                         # 2. Boundary Checks
#                         if win_start < 0:
#                             win_start = 0
#                             win_end = self.sample_length
                        
#                         if win_end > full_len:
#                             win_end = full_len
#                             win_start = max(0, full_len - self.sample_length)
                        
#                         # 3. Slice
#                         x = full_data[:, win_start:win_end]
                        
#                         # 4. Pad if file is smaller than window
#                         if x.shape[1] < self.sample_length:
#                             diff = self.sample_length - x.shape[1]
#                             x = np.pad(x, ((0,0), (0, diff)), mode='constant')

#                         # 5. Label (Adjust to 0-index)
#                         y = int(seg[self.labels_str]) - 1 if self.labels_str == "label_6way" else int(seg[self.labels_str])

#                         if self.channel_idx is not None: 
#                             if "cbramod" in self.model_type:
#                                 temp_arr = np.zeros_like(x)
#                                 temp_arr[self.channel_idx] = x[self.channel_idx]
#                                 x = temp_arr
#                             else:
#                                 x = x[self.channel_idx]
                        
#                         yield x, y
        
#         # Buffer Shuffle Logic
#         iterator = stream_chunks()
        
#         if self.shuffle:
#             buffer = []
#             try:
#                 for _ in range(self.buffer_size):
#                     buffer.append(next(iterator))
#             except StopIteration:
#                 pass 
            
#             for item in iterator:
#                 idx = rng.randint(0, len(buffer) - 1)
#                 yield buffer[idx]
#                 buffer[idx] = item
            
#             rng.shuffle(buffer)
#             for item in buffer:
#                 yield item
#         else:
#             yield from iterator


import lmdb
import pickle
import numpy as np 
import random 
import hashlib
import copy
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import IterableDataset, get_worker_info

class tuev_1s(IterableDataset):
    def __init__(
            self,
            lmdb_path: str=None,
            data_list: list=None,
            shuffle: bool=True,
            buffer_size: int=2000,
            seq_len: int=1,
            fs: int=200,
            percent_train_per_subject: float=None,
            n_samples_per_subject: int=None,
            seed: int=42,
            channel_idx: list=None,
            label_noise: float=None,
            num_classes: int=6,
            model_type: str=None,
            labels_str: str="label_6way",
            **kwargs
    ):
        
        super(tuev_1s, self).__init__()
        
        self.lmdb_path = lmdb_path
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.seq_len = seq_len
        self.fs = fs 
        self.sample_length = int(fs * seq_len)
        self.percent_train_per_subject = percent_train_per_subject
        self.n_samples_per_subject = n_samples_per_subject
        self.seed = seed
        self.channel_idx = channel_idx
        self.label_noise = label_noise
        self.num_classes = num_classes
        self.model_type = model_type
        self.labels_str = labels_str

        # -----------------------------------------------------------
        # 1. Group by Subject & Distribute Quota
        # -----------------------------------------------------------
        working_data_list = copy.deepcopy(data_list)
        self.data_list = []
        
        # Group by Subject ID (first part of the key)
        dict_subject_list = defaultdict(list)
        for d in working_data_list:
            subject = d["key"].split("_")[0]
            dict_subject_list[subject].append(d)

        if n_samples_per_subject is not None:
            for subject, files in dict_subject_list.items():
                files.sort(key=lambda x: x['key'])
                
                events_needed = n_samples_per_subject
                
                for f in files:
                    if events_needed <= 0:
                        continue
                    
                    # Note: f['length'] is provided by the metadata CSV
                    available_events = int(f['length'])
                    
                    quota = min(events_needed, available_events)
                    f['event_quota'] = quota
                    
                    self.data_list.append(f)
                    events_needed -= quota
        else:
            self.data_list = working_data_list

        # -----------------------------------------------------------
        # 2. Calculate Total Length
        # -----------------------------------------------------------
        self.length = 0
        mult_factor = 1 if percent_train_per_subject is None else percent_train_per_subject

        for d in self.data_list:
            if 'event_quota' in d:
                self.length += d['event_quota']
            else:
                self.length += int(d["length"] * mult_factor)

    def _get_worker_slice(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return self.data_list
        
        per_worker = int(np.ceil(len(self.data_list) / float(worker_info.num_workers)))
        worker_id = worker_info.id
        start = worker_id * per_worker
        end = min(start + per_worker, len(self.data_list))
        return self.data_list[start:end]
    
    def __len__(self):
        return self.length
    
    def __iter__(self):
        rng = random.Random(self.seed)
        worker_data_list = self._get_worker_slice()
        
        worker_data_list = list(worker_data_list)
        if self.shuffle:
            rng.shuffle(worker_data_list)

        env = lmdb.open(
            self.lmdb_path, 
            readonly=True, 
            lock=False, 
            readahead=False, 
            meminit=False
        )
        
        def stream_chunks():
            with env.begin() as txn:
                for item in worker_data_list:
                    key = item['key'].encode('ascii')
                    raw_bytes = txn.get(key)
                    if raw_bytes is None: continue

                    # FIX: Preprocessing now stores a LIST of event dicts directly
                    # Structure: [{"data": np.array, "label_6way": int, ...}, ...]
                    all_events = pickle.loads(raw_bytes) 
                    
                    n_total = len(all_events)
                    if n_total == 0: continue

                    # ------------------------------------------------
                    # DETERMINE SUBSET OF EVENTS TO PROCESS
                    # ------------------------------------------------
                    quota = item.get('event_quota', None)
                    
                    if quota is not None:
                        target_size = quota
                    elif self.percent_train_per_subject is not None:
                        target_size = int(n_total * self.percent_train_per_subject)
                    else:
                        target_size = n_total

                    # Perform Stratified Split if subsampling
                    keep_idxs = np.arange(n_total)
                    if target_size < n_total and target_size > 0:
                        labels = [s[self.labels_str] for s in all_events]
                        try:
                            keep_idxs, _ = train_test_split(
                                np.arange(n_total),
                                train_size=target_size,
                                stratify=labels,
                                random_state=self.seed 
                            )
                        except ValueError:
                            # Fallback if class too rare for stratification
                            keep_idxs, _ = train_test_split(
                                np.arange(n_total),
                                train_size=target_size,
                                random_state=self.seed
                            )
                        keep_idxs = sorted(keep_idxs)

                    # Select the subset of events
                    current_events = [all_events[i] for i in keep_idxs]
                    
                    # ------------------------------------------------
                    # NOISE INJECTION
                    # ------------------------------------------------
                    if self.label_noise is not None and self.label_noise > 0.0:
                        seed_str = f"{item['key']}_{self.seed}"
                        file_seed = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest(), 16) % (2**32)
                        file_rng = np.random.RandomState(file_seed)

                        for seg in current_events:
                            if file_rng.rand() < self.label_noise:
                                original = int(seg[self.labels_str])
                                candidates = [c for c in range(1, self.num_classes + 1) if c != original]
                                if candidates:
                                    seg[self.labels_str] = file_rng.choice(candidates)

                    # ------------------------------------------------
                    # DATA EXTRACTION
                    # ------------------------------------------------
                    if self.shuffle:
                        rng.shuffle(current_events)

                    for seg in current_events:
                        # 1. Get Data (Already extracted in preprocessing)
                        x = seg["data"]
                        # print(f"Shape of data: {x.shape}", flush=True)
                        # 2. Resize to self.sample_length
                        # Since we don't have the full file, we Crop (if too long) or Pad (if too short)
                        curr_len = x.shape[1]
                        
                        if curr_len > self.sample_length:
                            # Crop Center
                            start = (curr_len - self.sample_length) // 2
                            x = x[:, start : start + self.sample_length]
                        elif curr_len < self.sample_length:
                            # Pad End
                            diff = self.sample_length - curr_len
                            x = np.pad(x, ((0,0), (0, diff)), mode='constant')

                        # 3. Get Label
                        # Handle 0-indexing for 6-way (stored as 1-6)
                        # 4-way and 2-way are already 0-indexed in the revised preprocessing
                        raw_label = int(seg[self.labels_str])
                        if self.labels_str == "label_6way":
                            y = raw_label - 1
                        else:
                            y = raw_label

                        # 4. Channel Selection / Filtering
                        if self.channel_idx is not None: 
                            if self.model_type and "cbramod" in self.model_type:
                                temp_arr = np.zeros_like(x)
                                temp_arr[self.channel_idx] = x[self.channel_idx]
                                x = temp_arr
                            else:
                                x = x[self.channel_idx]
                        
                        yield x, y
        
        # Buffer Shuffle Logic
        iterator = stream_chunks()
        
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