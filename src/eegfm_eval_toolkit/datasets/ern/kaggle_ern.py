# Kaggle ERN dataset
# Perrin, M., Maby, E., Daligault, S., Bertrand, O., & Mattout, J. Objective and subjective evaluation of online error correction during P300-based spelling. Advances in Human-Computer Interaction, 2012, 4. (link)

import torch
import random
import math
import lmdb
import json
import pickle
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

class kaggle_ern(IterableDataset):
    def __init__(self, lmdb_path, data_list, shuffle=True, buffer_size=500, channel_idx=None, model_type: str=None):
        """
        Args:
            lmdb_path: Path to the LMDB directory.
            data_list: List of dicts containing {'key': ..., 'label': ...} from JSON splits.
            shuffle: Whether to shuffle files and use a shuffle buffer.
            buffer_size: Size of the reservoir for mixing trials from different subjects.
        """
        super(kaggle_ern, self).__init__()
        
        self.lmdb_path = lmdb_path
        self.data_list = data_list
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.channel_idx = channel_idx
        self.model_type = model_type

    def __len__(self):
        # Since ERN is pre-epoched, the number of chunks is simply the number of keys
        return len(self.data_list)

    def __iter__(self):
        # 1. SETUP: Open LMDB inside the worker process to be process-safe
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        
        # 2. WORKER SPLIT: Divide subjects/files among DataLoader workers
        worker_info = get_worker_info()
        if worker_info is None:
            my_list = self.data_list
        else:
            per_worker = int(math.ceil(len(self.data_list) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.data_list))
            my_list = self.data_list[iter_start:iter_end]

        # 3. TIER 1 SHUFFLE: Shuffle the entry list for this worker
        my_list = list(my_list)
        if self.shuffle:
            random.shuffle(my_list)

        # 4. GENERATOR: Yielding processed trials
        def stream_trials():
            with env.begin() as txn:
                for item in my_list:
                    key = item['key'].encode('ascii')
                    raw_bytes = txn.get(key)
                    
                    if raw_bytes is None:
                        continue
                    
                    # In our preprocessing, we saved individual 1.3s epochs
                    # shape is (Channels, Time)
                    x = pickle.loads(raw_bytes)
                    y = item['label']
                    
                    # Convert to torch tensor if needed, or keep as numpy for the buffer
                    if self.channel_idx is not None:
                        # if "cbramod" in self.model_type:
                        #     temp_arr = np.zeros_like(x)
                        #     temp_arr[self.channel_idx] = x[self.channel_idx]
                        #     x = temp_arr
                        # else:
                        x = x[self.channel_idx]
                            
                    yield torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

        # 5. TIER 2 SHUFFLE: The Shuffle Buffer (Reservoir Sampling)
        # Prevents the model from seeing all trials of Subject A, then all of Subject B.
        iterator = stream_trials()
        
        if self.shuffle:
            buffer = []
            try:
                # Initial buffer fill
                for _ in range(self.buffer_size):
                    buffer.append(next(iterator))
            except StopIteration:
                pass 
            
            for item in iterator:
                idx = random.randint(0, len(buffer) - 1)
                yield buffer[idx]
                buffer[idx] = item
            
            # Final flush of the buffer
            random.shuffle(buffer)
            for item in buffer:
                yield item
        else:
            yield from iterator

class kaggle_ern_map_style(torch.utils.data.Dataset):
    def __init__(self, lmdb_path, data_list):
        self.lmdb_path = lmdb_path
        self.data_list = data_list
        self.env = None # Opened lazily

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        
        item = self.data_list[idx]
        with self.env.begin() as txn:
            raw_bytes = txn.get(item['key'].encode('ascii'))
            x = pickle.loads(raw_bytes)
            y = item['label']
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)