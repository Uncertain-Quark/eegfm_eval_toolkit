from eegfm_eval_toolkit.datasets.motor_imagery.mmiphysionet import mmiphysionet_npz, mmiphysionet_reconstruction_npz
from eegfm_eval_toolkit.datasets.motor_imagery.bciciv_2a import bciciv_2a_npz, bciciv_2a_reconstruction_npz
from eegfm_eval_toolkit.datasets.normal_abnormal.tuab import tuab
from eegfm_eval_toolkit.datasets.events.tuev import tuev
from eegfm_eval_toolkit.datasets.events.tuev_1s import tuev_1s
from eegfm_eval_toolkit.datasets.mental_health.mdd_mal import mdd_mal
from eegfm_eval_toolkit.datasets.sleep.sleep_edfx import sleep_edf
from eegfm_eval_toolkit.datasets.ern.errp_hri import errp_hri
from eegfm_eval_toolkit.datasets.ern.kaggle_ern import kaggle_ern
from eegfm_eval_toolkit.datasets.eeg_rtmri.eeg_rtmri import eeg_rtmri
from eegfm_eval_toolkit.datasets.p300.physionetp300 import physionetp300
from eegfm_eval_toolkit.datasets.tueg.tueg import get_tueg_eeg_dataloader

from eegfm_eval_toolkit.utils.make_channels_config import ChannelSampler

from functools import partial
from collections import defaultdict
import os, json, glob
import numpy as np
import random
import pandas as pd 
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, random_split, Subset

def make_kaggle_ern_dataloaders(
        mode: str="2class",
        return_datasets: bool=False,
        debug: bool=False,
        percent_train_subjects: float=None,
        n_subjects: int=None,
        percent_train_per_subject: float=None,
        n_samples_per_subject: int=None,
        feature: str="raw_norm",
        fs: int=200,
        batch_size: int=32,
        buffer_size: int=500,
        seed: int=42,
        channel_config: dict=None,
        model_type: str=None,
        **kwargs
):
    
    rng = random.Random(seed)
    split_data_path = os.path.join(os.getenv("eegfm_eval_toolkit_SPLIT_PATH", "./splits"), "kaggle_ern")
    split_paths = sorted(glob.glob(split_data_path + f"/{mode}*.json"))
    
    data_path = os.path.join(os.getenv("eegfm_eval_toolkit_PREPROCESSED_DATA_ROOT", "./preprocessed"), f"kaggle_ern_{feature}_{fs}")

    # get the list of channels
    channels_path = os.path.join(data_path, "channels.json")
    channels_names = json.load(open(channels_path, "r")) if os.path.exists(channels_path) else None

    channel_idx = None
    if channel_config is not None:
        channel_sampler = ChannelSampler(preprocessed_root=data_path, seed=seed)
        channel_idx = channel_sampler.process_config(channel_config)

    for i, split_path in enumerate(split_paths):
        print(f"Processing Fold: {i} {split_path}")

        metadata_info = json.load(open(split_path, "r"))
        train_meta = metadata_info["train"]
        train_subjects = [d["key"].split("_")[0] for d in train_meta]

        if (percent_train_subjects is not None and percent_train_subjects < 1.0) or (n_subjects is not None):
            # Extract unique subjects
            unique_subjects = list(set(train_subjects))
            rng.shuffle(unique_subjects)
            
            # Calculate how many subjects to keep
            n_keep = int(len(unique_subjects) * percent_train_subjects) if percent_train_subjects is not None else n_subjects
            n_keep = max(1, n_keep) # Ensure at least 1 subject is kept
            
            kept_subjects = set(unique_subjects[:n_keep])
            print(f"  [Filter] Keeping {len(kept_subjects)}/{len(unique_subjects)} subjects.")
            
            # Filter the metadata list
            train_meta = [d for d in train_meta if d["key"].split("_")[0] in kept_subjects]

        if (percent_train_per_subject is not None and percent_train_per_subject < 1.0) or n_samples_per_subject is not None:
            filtered_train_meta = []
            
            # Group data by subject first
            subj_map = defaultdict(list)
            for d in train_meta:
                subj_map[d["key"].split("_")[0]].append(d)

            # samples_per_subject_count = None
            # if n_samples is not None and len(subj_map) > 0:
            #     samples_per_subject_count = max(1, n_samples // len(subj_map))
            #     print(f"  [Info] n_samples={n_samples} over {len(subj_map)} subjects results in {samples_per_subject_count} samples/subject.")

            for subj, items in subj_map.items():
                labels = [d["label"] for d in items]

                if n_samples_per_subject is not None:
                    train_size_arg = min(n_samples_per_subject, len(labels))
                    # If subject has fewer samples than requested, take them all
                    if len(items) <= train_size_arg:
                        filtered_train_meta.extend(items)
                        continue
                else:
                    train_size_arg = percent_train_per_subject
                    # Skip stratification if too few samples for percentage split
                    if len(items) < 2:
                        if percent_train_per_subject >= 0.5:
                            filtered_train_meta.extend(items)
                        continue

                try:
                    # Stratified split
                    # train_test_split handles both float (0.0-1.0) and int (absolute number)
                    kept_items, _ = train_test_split(
                        items,
                        train_size=train_size_arg,
                        stratify=labels,
                        random_state=seed
                    )
                except ValueError:
                    # Fallback: Random split if stratification fails (e.g. rare class)
                    kept_items, _ = train_test_split(
                        items,
                        train_size=train_size_arg,
                        random_state=seed
                    )
                
                filtered_train_meta.extend(kept_items)
            
            print(f"  [Filter] Reduced samples from {len(train_meta)} to {len(filtered_train_meta)} ")
                #   f"({percent_train_per_subject*100}% per subject).")
            train_meta = filtered_train_meta

        train_dataset = kaggle_ern(data_path, train_meta, shuffle=True, buffer_size=buffer_size, channel_idx=channel_idx, model_type=model_type)
        val_dataset = kaggle_ern(data_path, metadata_info["val"], shuffle=False, buffer_size=buffer_size, channel_idx=channel_idx, model_type=model_type)
        test_dataset = kaggle_ern(data_path, metadata_info["test"], shuffle=False, buffer_size=buffer_size, channel_idx=channel_idx, model_type=model_type)

        if return_datasets:
            yield i, channel_idx, channels_names, train_dataset, val_dataset, test_dataset
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        yield i, channel_idx, channels_names, train_dataloader, val_dataloader, test_dataloader

def make_sleep_edfx_dataloaders(
        mode: str="5class_cassette",
        return_datasets: bool=False,
        debug: bool=False,
        percent_train_subjects: float=None,
        n_subjects: int=None,
        percent_train_per_subject: float=None,
        n_samples_per_subject: int=None,
        feature: str="raw_norm",
        fs: int=100,
        batch_size: int=32,
        buffer_size: int=1000,
        seed: int=42,
        channel_config: dict=None,
        model_type: str=None,
        eval_batch_size: int=None,
        task_type: str="sleep",
        **kwargs
):
    
    rng = random.Random(seed)
    split_data_path = os.path.join(os.getenv("eegfm_eval_toolkit_SPLIT_PATH", "./splits"), "sleep_edfx")
    split_paths = sorted(glob.glob(split_data_path + f"/{mode}*.json"))
    
    data_path = os.path.join(os.getenv("eegfm_eval_toolkit_PREPROCESSED_DATA_ROOT", "./preprocessed"), f"sleep_edfx_{feature}_{fs}")

    channels_path = os.path.join(data_path, "channels.json")
    channels_names = json.load(open(channels_path, "r")) if os.path.exists(channels_path) else None

    channel_idx = None
    if channel_config is not None:
        channel_sampler = ChannelSampler(preprocessed_root=data_path, seed=seed)
        channel_idx = channel_sampler.process_config(channel_config)

    for i, split_path in enumerate(split_paths):
        print(f"Processing Fold: {i} {split_path}")

        metadata_info = json.load(open(split_path, "r"))

        train_meta = metadata_info["train"]

        if (percent_train_subjects is not None and percent_train_subjects < 1.0) or n_subjects is not None:
            # Extract unique subjects
            unique_subjects = list(set(d["subject"] for d in train_meta))
            rng.shuffle(unique_subjects)
            
            # Calculate how many subjects to keep
            n_keep = int(len(unique_subjects) * percent_train_subjects) if percent_train_subjects is not None else n_subjects
            n_keep = max(1, n_keep) # Ensure at least 1 subject is kept
            
            kept_subjects = set(unique_subjects[:n_keep])
            print(f"  [Filter] Keeping {len(kept_subjects)}/{len(unique_subjects)} subjects.")
            
            # Filter the metadata list
            train_meta = [d for d in train_meta if d["subject"] in kept_subjects]

        train_dataset = sleep_edf(data_path, train_meta, shuffle=True, buffer_size=buffer_size, percent_train_per_subject=percent_train_per_subject, 
                                  n_samples_per_subject=n_samples_per_subject, seed=seed, channel_idx=channel_idx, model_type=model_type, task_type=task_type)
        val_dataset = sleep_edf(data_path, metadata_info["val"], shuffle=False, buffer_size=buffer_size, channel_idx=channel_idx, model_type=model_type, task_type=task_type)
        test_dataset = sleep_edf(data_path, metadata_info["test"], shuffle=False, buffer_size=buffer_size, channel_idx=channel_idx, model_type=model_type, task_type=task_type)

        if return_datasets:
            yield i, channel_idx, channels_names, train_dataset, val_dataset, test_dataset
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size if eval_batch_size is None else eval_batch_size, shuffle=False, num_workers=16)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size if eval_batch_size is None else eval_batch_size, shuffle=False, num_workers=16)

        yield i, channel_idx, channels_names, train_dataloader, val_dataloader, test_dataloader

def make_mdd_mal_dataloaders(
        mode: str="EC",
        return_datasets: bool=False,
        debug: bool=False,
        percent_train_subjects: float=None,
        n_subjects: int=None,
        percent_train_per_subject: float=None,
        n_samples_per_subject: int=None,
        feature: str="raw_norm",
        fs: int=256,
        n_channels: int=19,
        seq_len: int=10,
        batch_size: int=32,
        seed: int=42,
        channel_config: dict=None,
        model_type: str=None,
        norm_type: str=None,
        **kwargs
):
    rng = random.Random(seed)
    split_data_path = os.path.join(os.getenv("eegfm_eval_toolkit_SPLIT_PATH", "./splits"), "mdd_mal")
    split_paths = sorted(glob.glob(split_data_path + f"/*{mode}*.json"))
    
    data_path_postfix = f"mdd_mal_{feature}_{fs}"
    if norm_type is not None: data_path_postfix += f"_{norm_type}"
    data_path = os.path.join(os.getenv("eegfm_eval_toolkit_PREPROCESSED_DATA_ROOT", "./preprocessed"), data_path_postfix)

    channels_path = os.path.join(data_path, "channels.json")
    channels_names = json.load(open(channels_path, "r")) if os.path.exists(channels_path) else None

    channel_idx = None
    if channel_config is not None:
        channel_sampler = ChannelSampler(preprocessed_root=data_path, seed=seed)
        channel_idx = channel_sampler.process_config(channel_config)

    for i, split_path in enumerate(split_paths):
        print(f"Processing Fold: {i} {split_path}")

        metadata_info = json.load(open(split_path, "r"))

        train_meta = metadata_info["train"]

        subjects_labels_dict = {}
        if (percent_train_subjects is not None and percent_train_subjects < 1.0) or n_subjects is not None:
            # Extract unique subjects
            unique_subjects = list(set(d["subject"] for d in train_meta))
            for d in train_meta:
                if d["subject"] not in subjects_labels_dict.keys(): subjects_labels_dict[d["subject"]] = d["label"]
            rng.shuffle(unique_subjects)

            unique_subjects_labels = [subjects_labels_dict[s] for s in unique_subjects]
            
            # Calculate how many subjects to keep
            n_keep = int(len(unique_subjects) * percent_train_subjects) if percent_train_subjects is not None else n_subjects
            n_keep = max(2, n_keep) # Ensure at least 2 subjects are kept in order to have one depressed and one control for classification
            
            # kept_subjects = set(unique_subjects[:n_keep])
            # compute the kept subjects to stratify for MDD and Control
            kept_subjects, _ = train_test_split(unique_subjects, train_size=n_keep, stratify=unique_subjects_labels)
            print(f"  [Filter] Keeping {len(kept_subjects)}/{len(unique_subjects)} subjects.")
            
            # Filter the metadata list
            train_meta = [d for d in train_meta if d["subject"] in kept_subjects]

        train_dataset = mdd_mal(data_path, train_meta, seq_len_sec=seq_len, fs=fs, percent_train_per_subject=percent_train_per_subject, 
                                n_samples_per_subject=n_samples_per_subject, seed=seed, channel_idx=channel_idx, model_type=model_type)
        val_dataset = mdd_mal(data_path, metadata_info["val"], seq_len_sec=seq_len, fs=fs, channel_idx=channel_idx, model_type=model_type)
        test_dataset = mdd_mal(data_path, metadata_info["test"], seq_len_sec=seq_len, fs=fs, channel_idx=channel_idx, model_type=model_type)

        if return_datasets:
            yield i, channel_idx, channels_names, train_dataset, val_dataset, test_dataset
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        yield i, channel_idx, channels_names, train_dataloader, val_dataloader, test_dataloader

def make_tuev_dataloaders(
    mode: str="cv",
    return_datasets: bool=False,
    debug: bool=False,
    percent_train_subjects: float=None,
    n_subjects: int=None,
    percent_train_per_subject: float=None,
    n_samples_per_subject: int=None,
    feature: str="raw",
    fs: int=200,
    seq_len: int=5,
    batch_size: int=32,
    seed: int=42,
    channel_config: dict=None,
    model_type: str=None,
    **kwargs   
):
    rng = random.Random(seed)

    split_data_path = os.path.join(os.getenv("eegfm_eval_toolkit_SPLIT_PATH", "./splits"), "tuev")
    print(f"split path: {split_data_path}")
    split_paths = sorted(glob.glob(split_data_path + f"/{mode}*.json"))

    data_path = os.path.join(os.getenv("eegfm_eval_toolkit_PREPROCESSED_DATA_ROOT", "./preprocessed"), f"tuev_{feature}_{fs}")

    channels_path = os.path.join(data_path, "channels.json")
    channels_names = json.load(open(channels_path, "r")) if os.path.exists(channels_path) else None

    channel_idx = None
    if channel_config is not None:
        channel_sampler = ChannelSampler(preprocessed_root=data_path, seed=seed)
        channel_idx = channel_sampler.process_config(channel_config)
        
    for i, split_path in enumerate(split_paths):
        print(f"Processing fold: {i} {split_path}")

        metadata_info = json.load(open(split_path, "r"))

        train_meta = metadata_info["train"]

        if (percent_train_subjects is not None and percent_train_subjects < 1.0) or n_subjects is not None:
            # Extract unique subjects
            unique_subjects = list(set(d["key"].split("_")[0] for d in train_meta))
            rng.shuffle(unique_subjects)
            
            # Calculate how many subjects to keep
            n_keep = int(len(unique_subjects) * percent_train_subjects) if percent_train_subjects is not None else n_subjects
            n_keep = max(1, n_keep) # Ensure at least 1 subject is kept
            
            kept_subjects = set(unique_subjects[:n_keep])
            print(f"  [Filter] Keeping {len(kept_subjects)}/{len(unique_subjects)} subjects.")
            
            # Filter the metadata list
            train_meta = [d for d in train_meta if d["key"].split("_")[0] in kept_subjects]

        train_dataset = tuev(lmdb_path=data_path, data_list=train_meta, shuffle=True, seq_len=seq_len, 
                             percent_train_per_subject=percent_train_per_subject, n_samples_per_subject=n_samples_per_subject, 
                             seed=seed, channel_idx=channel_idx, model_type=model_type)
        val_dataset = tuev(lmdb_path=data_path, data_list=metadata_info["val"], shuffle=False, seq_len=seq_len, 
                           channel_idx=channel_idx, model_type=model_type)
        test_dataset = tuev(lmdb_path=data_path, data_list=metadata_info["test"], shuffle=False, seq_len=seq_len, channel_idx=channel_idx,
                            model_type=model_type)

        if return_datasets:
            yield i, channel_idx, channels_names, train_dataset, val_dataset, test_dataset
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        yield i, channel_idx, channels_names, train_dataloader, val_dataloader, test_dataloader

def make_tuev_1s_dataloaders(
    mode: str="cv",
    return_datasets: bool=False,
    debug: bool=False,
    percent_train_subjects: float=None,
    n_subjects: int=None,
    percent_train_per_subject: float=None,
    n_samples_per_subject: int=None,
    feature: str="raw",
    fs: int=200,
    seq_len: int=1,
    batch_size: int=32,
    seed: int=42,
    channel_config: dict=None,
    model_type: str=None,
    labels_str: str="label_6way",
    **kwargs   
):
    rng = random.Random(seed)

    split_data_path = os.path.join(os.getenv("eegfm_eval_toolkit_SPLIT_PATH", "./splits"), "tuev_1s")
    print(f"split path: {split_data_path}")
    split_paths = sorted(glob.glob(split_data_path + f"/{mode}*.json"))

    data_path = os.path.join(os.getenv("eegfm_eval_toolkit_PREPROCESSED_DATA_ROOT", "./preprocessed"), f"tuev_1s_{feature}_{fs}")

    channels_path = os.path.join(data_path, "channels.json")
    channels_names = json.load(open(channels_path, "r")) if os.path.exists(channels_path) else None

    channel_idx = None
    if channel_config is not None:
        channel_sampler = ChannelSampler(preprocessed_root=data_path, seed=seed)
        channel_idx = channel_sampler.process_config(channel_config)
        
    for i, split_path in enumerate(split_paths):
        print(f"Processing fold: {i} {split_path}")

        metadata_info = json.load(open(split_path, "r"))

        train_meta = metadata_info["train"]

        if (percent_train_subjects is not None and percent_train_subjects < 1.0) or n_subjects is not None:
            # Extract unique subjects
            unique_subjects = list(set(d["key"].split("_")[0] for d in train_meta))
            rng.shuffle(unique_subjects)
            
            # Calculate how many subjects to keep
            n_keep = int(len(unique_subjects) * percent_train_subjects) if percent_train_subjects is not None else n_subjects
            n_keep = max(1, n_keep) # Ensure at least 1 subject is kept
            
            kept_subjects = set(unique_subjects[:n_keep])
            print(f"  [Filter] Keeping {len(kept_subjects)}/{len(unique_subjects)} subjects.")
            
            # Filter the metadata list
            train_meta = [d for d in train_meta if d["key"].split("_")[0] in kept_subjects]

        train_dataset = tuev_1s(lmdb_path=data_path, data_list=train_meta, shuffle=True, seq_len=seq_len, 
                             percent_train_per_subject=percent_train_per_subject, n_samples_per_subject=n_samples_per_subject, 
                             seed=seed, channel_idx=channel_idx, model_type=model_type, labels_str=labels_str)
        val_dataset = tuev_1s(lmdb_path=data_path, data_list=metadata_info["val"], shuffle=False, seq_len=seq_len, 
                           channel_idx=channel_idx, model_type=model_type, labels_str=labels_str)
        test_dataset = tuev_1s(lmdb_path=data_path, data_list=metadata_info["test"], shuffle=False, seq_len=seq_len, channel_idx=channel_idx,
                            model_type=model_type, labels_str=labels_str)

        if return_datasets:
            yield i, channel_idx, channels_names, train_dataset, val_dataset, test_dataset
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

        yield i, channel_idx, channels_names, train_dataloader, val_dataloader, test_dataloader

def make_bciciv_2a_dataloaders(
        mode: str="4class_cross_subject",
        return_datasets: bool=False,
        debug: bool=False,
        percent_train_subjects: float=None,
        n_subjects: int=None,
        percent_train_per_subject: float=None,
        n_samples_per_subject: int=None,
        feature: str="raw_norm",
        fs: int=250,
        batch_size: int=64,
        global_dataset_info: dict=None,
        seed: int=42,
        channel_config: dict=None,
        label_noise: float=None,
        aug_dict: dict=None,
        model_type: str=None,
        task_type: str="motor",
        is_reconstruction: bool=False,
        is_adversarial_training: bool=False,
        **kwargs
):
    rng = random.Random(seed)
    split_data_path = os.path.join(os.getenv("eegfm_eval_toolkit_SPLIT_PATH", "./splits"), "bciciv_2a")
    print(f"split path: {split_data_path}")
    split_paths = sorted(glob.glob(split_data_path + f"/{mode}_fold*.json"))

    # if channel_config is set, get the channel indices to sample
    data_path = os.path.join(os.getenv("eegfm_eval_toolkit_PREPROCESSED_DATA_ROOT", "./preprocessed"), f"bciciv_2a_{feature}_{fs}")
    channels_path = os.path.join(data_path, "channels.json")
    channels_names = json.load(open(channels_path, "r")) if os.path.exists(channels_path) else None

    channel_idx = None
    if channel_config is not None:
        channel_sampler = ChannelSampler(preprocessed_root=os.path.join(os.getenv("eegfm_eval_toolkit_PREPROCESSED_DATA_ROOT", "./preprocessed"), f"bciciv_2a_{feature}_{fs}"), seed=seed)
        channel_idx = channel_sampler.process_config(channel_config)

    if global_dataset_info is None:
        data_path = os.path.join(os.getenv("eegfm_eval_toolkit_PREPROCESSED_DATA_ROOT", "./preprocessed"), f"bciciv_2a_{feature}_{fs}", "data.npz")
        global_dataset = bciciv_2a_npz(data_path=data_path, channel_idx=channel_idx, label_noise=label_noise, aug_dict=aug_dict, model_type=model_type, task_type=task_type, is_adversarial_training=is_adversarial_training) if not is_reconstruction else bciciv_2a_reconstruction_npz(data_path=data_path, channel_idx=channel_idx, label_noise=label_noise, aug_dict=aug_dict, model_type=model_type, task_type=task_type, is_adversarial_training=is_adversarial_training)
    else:
        global_dataset = global_dataset_info["dataset"]
        global_dataset.channel_idx = channel_idx
        global_dataset.aug_dict = aug_dict
    
    for i, split_path in enumerate(split_paths):

        print(f"Processing Fold: {i} {split_path}")

        metadata_info = json.load(open(split_path, "r"))
        
        train_keys = metadata_info["train"]
        val_keys = metadata_info["val"]
        test_keys = metadata_info["test"]

        if (percent_train_subjects is not None and percent_train_subjects < 1.0) or n_subjects is not None:
            # Identify all unique subjects in the current training split
            # Key format assumed: "S001_C0_..." -> split("_")[0] gives "S001"
            unique_subjects = sorted(list(set(k.split("_")[0] for k in train_keys)))
            
            n_keep = max(1, int(len(unique_subjects) * percent_train_subjects)) if percent_train_subjects is not None else n_subjects

            # 3. Select the subjects
            # We use a fixed seed here to ensure that if you re-run the same experiment 
            # with the same params, you get the same subset of subjects.
            rng.shuffle(unique_subjects)
            selected_subjects = set(unique_subjects[:n_keep])

            # 4. Filter keys
            # Only keep keys that start with one of the selected subject IDs
            train_keys = [k for k in train_keys if k.split("_")[0] in selected_subjects]

            print(f"  > Subsampling Training Subjects: {len(unique_subjects)} -> {len(selected_subjects)} "
                  f"({(len(selected_subjects)/len(unique_subjects))*100:.1f}%) | "
                  f"Remaining Trials: {len(train_keys)}")

        if (percent_train_per_subject is not None and percent_train_per_subject < 1.0) or n_samples_per_subject is not None:
            # 1. Parse metadata to group by subject
            # We use a dictionary to hold list of (key, label) tuples for each subject
            from collections import defaultdict
            from sklearn.model_selection import train_test_split

            subj_groups = defaultdict(list)
            
            for key in train_keys:
                # Key format: "Subject_Session_Label_..." (Adjust index if needed based on your key format)
                # Based on your snippet: split_str[0] is Subject, split_str[2] is Label
                parts = key.split("_")
                subj_id = parts[0]
                label = parts[2]
                subj_groups[subj_id].append((key, label))

            filtered_keys = []

            # 2. Iterate through each subject and subsample
            for subj_id, items in subj_groups.items():
                s_keys = [x[0] for x in items]
                s_labels = [x[1] for x in items]
                
                # Safety check: Need at least 2 items to split
                if len(s_keys) < 2:
                    # If dataset is extremely small, just keep the item to avoid crashing
                    filtered_keys.extend(s_keys)
                    continue
                
                if n_samples_per_subject is not None:
                    train_arg_size = n_samples_per_subject
                else:
                    train_arg_size = percent_train_per_subject

                try:
                    # Stratified split: Keeps 'percent' of this subject's data while maintaining label ratio
                    keep_keys, _ = train_test_split(
                        s_keys,
                        train_size=train_arg_size,
                        stratify=s_labels,
                        random_state=seed  # Fixed seed for reproducibility
                    )
                except ValueError:
                    # Fallback: If a class has too few samples to stratify (e.g. 1 sample of class X),
                    # fall back to random splitting without stratification.
                    keep_keys, _ = train_test_split(
                        s_keys,
                        train_size=train_arg_size,
                        random_state=seed
                    )
                
                filtered_keys.extend(keep_keys)

            # 3. Update the main list
            print(f"  > Stratified Per-Subject: {len(train_keys)} -> {len(filtered_keys)} trials ")
                #   f"({percent_train_per_subject*100}% per subject).")
            train_keys = filtered_keys

        train_indices = [global_dataset.key_to_idx[k] for k in train_keys if k in global_dataset.key_to_idx]
        val_indices = [global_dataset.key_to_idx[k] for k in val_keys if k in global_dataset.key_to_idx]
        test_indices = [global_dataset.key_to_idx[k] for k in test_keys if k in global_dataset.key_to_idx]

       
        if task_type == "subject":
            # 1. Create a consistent 0-indexed mapping for all subjects present in THIS fold
            # This handles cases where you subsampled and dropped subjects entirely
            all_fold_keys = train_keys + val_keys + test_keys
            unique_subjects = sorted(list(set(int(k.split('_')[0][1:]) for k in all_fold_keys)))
            print(f"Fold {i} Unique Subjects: {unique_subjects}")
            
            sub_to_idx = {sub: idx for idx, sub in enumerate(unique_subjects)}

            # 2. Use the custom wrapper to map labels on the fly
            train_dataset = CustomLabelSubset(global_dataset, train_indices, train_keys, sub_to_idx)
            val_dataset = CustomLabelSubset(global_dataset, val_indices, val_keys, sub_to_idx)
            test_dataset = CustomLabelSubset(global_dataset, test_indices, test_keys, sub_to_idx)
        else:
            # Standard PyTorch Subset for motor tasks
            train_dataset = Subset(global_dataset, train_indices)
            val_dataset = Subset(global_dataset, val_indices)
            test_dataset = Subset(global_dataset, test_indices)
        
        if return_datasets:
            yield i, channel_idx, channels_names, train_dataset, val_dataset, test_dataset

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        yield i, channel_idx, channels_names, train_loader, val_loader, test_loader


def make_mmiphysionet_dataloaders(
        mode: str="left_right",
        return_datasets: bool=False,
        debug: bool=False,
        percent_train_subjects: float=None,
        n_subjects: int=None,
        percent_train_per_subject: float=None,
        n_samples_per_subject: int=None,
        feature: str="raw_norm",
        fs: int=160,
        batch_size: int=64,
        global_dataset_info: dict=None,
        seed: int=42,
        channel_config: dict=None,
        model_type: str=None,
        dataset_name: str="mmiphysionet",
        task_type: str="motor",
        **kwargs
):
    
    rng = random.Random(seed)
    split_data_path = os.path.join(os.getenv("eegfm_eval_toolkit_SPLIT_PATH", "./splits"), dataset_name)
    print(f"split path: {split_data_path}")
    split_paths = sorted(glob.glob(split_data_path + f"/{mode}*.json"))

    # channel sampler
    data_path = os.path.join(os.getenv("eegfm_eval_toolkit_PREPROCESSED_DATA_ROOT", "./preprocessed"), f"{dataset_name}_{feature}_{fs}")
    channels_path = os.path.join(data_path, "channels.json")
    channels_names = json.load(open(channels_path, "r")) if os.path.exists(channels_path) else None
    
    channel_idx = None
    if channel_config is not None:
        channel_sampler = ChannelSampler(preprocessed_root=os.path.join(os.getenv("eegfm_eval_toolkit_PREPROCESSED_DATA_ROOT", "./preprocessed"), f"{dataset_name}_{feature}_{fs}"), seed=seed)
        channel_idx = channel_sampler.process_config(channel_config)

    if global_dataset_info is None:
        data_path = os.path.join(os.getenv("eegfm_eval_toolkit_PREPROCESSED_DATA_ROOT", "./preprocessed"), f"{dataset_name}_{feature}_{fs}", "data.npz")
        global_dataset = mmiphysionet_npz(data_path=data_path, channel_idx=channel_idx, model_type=model_type, task_type=task_type)
    else:
        global_dataset = global_dataset_info["dataset"]
        global_dataset.channel_idx = channel_idx

    for i, split_path in enumerate(split_paths):
        print(f"Feature: {feature}", flush=True)
        print(f"Processing Fold: {i} {split_path}")

        metadata_info = json.load(open(split_path, "r"))
        
        train_keys = metadata_info["train"]
        val_keys = metadata_info["val"]
        test_keys = metadata_info["test"]

        # --- PERCENT TRAIN SUBJECTS LOGIC ---
        if percent_train_subjects is not None or n_subjects is not None:
            # 1. Identify all unique subjects in the current training split
            # Key format assumed: "S001_C0_..." -> split("_")[0] gives "S001"
            unique_subjects = sorted(list(set(k.split("_")[0] for k in train_keys)))
            
            # 2. Determine target number of subjects
            n_keep = int(len(unique_subjects) * percent_train_subjects) if percent_train_subjects is not None else n_subjects
            
            # Ensure we don't ask for more than we have
            n_keep = max(1, min(n_keep, len(unique_subjects)))

            # 3. Select the subjects
            # We use a fixed seed here to ensure that if you re-run the same experiment 
            # with the same params, you get the same subset of subjects.
            rng.shuffle(unique_subjects)
            selected_subjects = set(unique_subjects[:n_keep])
            print(f"Selected Subject: {selected_subjects}")

            # 4. Filter keys
            # Only keep keys that start with one of the selected subject IDs
            train_keys = [k for k in train_keys if k.split("_")[0] in selected_subjects]

            print(f"  > Subsampling Training Subjects: {len(unique_subjects)} -> {len(selected_subjects)} "
                  f"({(len(selected_subjects)/len(unique_subjects))*100:.1f}%) | "
                  f"Remaining Trials: {len(train_keys)}")
        # ------------------------------------

        # --- B. PERCENT TRAIN PER SUBJECT (PER CLASS) LOGIC ---
        if percent_train_per_subject is not None or n_samples_per_subject is not None:
            
            # 1. Group keys by (Subject, Class)
            # Structure: grouped_keys["S001"]["C0"] = [list of keys]
            grouped_keys = defaultdict(lambda: defaultdict(list))
            
            for key in train_keys:
                parts = key.split("_") # ['S001', 'C0', '03', '001']
                sub_id = parts[0]
                class_id = parts[1]
                grouped_keys[sub_id][class_id].append(key)
            print(f"Number of subjects in grouped keys: {len(grouped_keys)})")

            new_train_keys = []

            # 2. Iterate and subsample
            total_before = len(train_keys)
            
            for sub_id in grouped_keys:
                # ensure that the number of samples per class is evenly divided
                n_samples_per_subject_class = n_samples_per_subject//len(grouped_keys[sub_id])
                for class_id in grouped_keys[sub_id]:
                    trials = grouped_keys[sub_id][class_id]
                    
                    # Calculate how many to keep for this specific bucket
                    n_keep_trial = int(len(trials) * percent_train_per_subject) if percent_train_per_subject is not None else n_samples_per_subject_class
                    n_keep_trial = max(1, n_keep_trial) # Ensure at least 1 sample if possible
                    
                    # Shuffle and slice
                    # Note: We sort first to ensure rng.shuffle produces identical results 
                    # regardless of original list order
                    trials.sort() 
                    rng.shuffle(trials)
                    
                    selected_trials = trials[:n_keep_trial]
                    new_train_keys.extend(selected_trials)
            
            train_keys = new_train_keys
            if percent_train_per_subject is not None:
                print(f"  > Subsampling Trials per Subject ({percent_train_per_subject*100}%): "
                    f"{total_before} -> {len(train_keys)} trials")
            else:
                print(f" > Subsampling Trials per Subject: {n_samples_per_subject}: "
                      f"{total_before} -> {len(train_keys)} trials")
        # -----------------------------------------------------

        train_indices = [global_dataset.key_to_idx[k] for k in train_keys if k in global_dataset.key_to_idx]
        print(f"Length of train indices: {len(train_indices)}")
        val_indices = [global_dataset.key_to_idx[k] for k in val_keys if k in global_dataset.key_to_idx]
        test_indices = [global_dataset.key_to_idx[k] for k in test_keys if k in global_dataset.key_to_idx]

        train_dataset = Subset(global_dataset, train_indices)
        val_dataset = Subset(global_dataset, val_indices)
        test_dataset = Subset(global_dataset, test_indices)
        
        if return_datasets:
            return  i, channel_idx, channels_names, train_dataset, val_dataset, test_dataset

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        yield i, channel_idx, channels_names, train_loader, val_loader, test_loader


DATALOADERS_DICT={
        "cbramod_mmiphysionet_4class_cross_subject": partial(make_mmiphysionet_dataloaders, mode="4class_cross_subject"),
        "mmiphysionet_4class_cross_subject": partial(make_mmiphysionet_dataloaders, mode="4class_cross_subject"),
        "mmiphysionet_4class_within_subject": partial(make_mmiphysionet_dataloaders, mode="4class_within_subject"),
        "mmiphysionet_109_4class_cbramod": partial(make_mmiphysionet_dataloaders, mode="4class_cbramod", dataset_name="mmiphysionet_109"),
        "bciciv_2a_4class_cross_subject": partial(make_bciciv_2a_dataloaders, mode="4class_cross_subject"),
        "bciciv_2a_4class_subject_specific": partial(make_bciciv_2a_dataloaders, mode="4class_subject_specific"),
        "bciciv_2a_4class_subject_specific_te": partial(make_bciciv_2a_dataloaders, mode="4class_subject_specific_te"),
        "bciciv_2a_4class_within_subject": partial(make_bciciv_2a_dataloaders, mode="4class_within_subject"),
        "bciciv_2a_4class_within_subject_id": partial(make_bciciv_2a_dataloaders, mode="4class_within_subject_id", task_type="subject"),
        "bciciv_2a_4class_within_subject_motor": partial(make_bciciv_2a_dataloaders, mode="4class_within_subject_motor"),
        # Autoencoder with BCIC IV 2A dataset
        "bciciv_2a_autoencoder_within_4class": partial(make_bciciv_2a_dataloaders, is_reconstruction=True, mode="4class_within_subject"),
        "bciciv_2a_autoencoder_within_4class_adversarial": partial(make_bciciv_2a_dataloaders, is_reconstruction=True, mode="4class_within_subject", is_adversarial_training=True, task_type="subject"),

        "mmiphysionet_autoencoder_4class": partial(make_mmiphysionet_dataloaders, mode="4class_within_subject"),
        "bciciv_2a_autoencoder_within_4class_classification_motor": partial(make_bciciv_2a_dataloaders, mode="4class_within_subject_motor"),
        "bciciv_2a_autoencoder_within_4class_classification_id": partial(make_bciciv_2a_dataloaders, mode="4class_within_subject_id", task_type="subject"),
        "bciciv_2a_autoencoder_within_4class_classification_motor_all": partial(make_bciciv_2a_dataloaders, mode="4class_within_subject"),
        "bciciv_2a_autoencoder_within_4class_classification_id_all": partial(make_bciciv_2a_dataloaders, mode="4class_within_subject", task_type="subject"),
        "mmiphysionet_4class_within_subject_motor": partial(make_mmiphysionet_dataloaders, mode="4class_within_subject_motor"),
        "mmiphysionet_4class_within_subject_id": partial(make_mmiphysionet_dataloaders, mode="4class_within_subject_id", task_type="subject"),
        "bciciv_2a_4class_cross_subject_sess_transfer": partial(make_bciciv_2a_dataloaders, mode="4class_cross_subject_sess_transfer"),
        "mdd_mal_EC": partial(make_mdd_mal_dataloaders, mode="EC"),
        "mdd_mal_EO": partial(make_mdd_mal_dataloaders, mode="EO"),
        "sleep_edfx_5class_cassette": partial(make_sleep_edfx_dataloaders, mode="5class_cassette"),
        "sleep_edfx_5class_telemetry": partial(make_sleep_edfx_dataloaders, mode="5class_telemetry"),
        "kaggle_ern_2class": partial(make_kaggle_ern_dataloaders, mode="2class"),
        "tuev_cv": partial(make_tuev_dataloaders, mode="cv"),
        "tuev_1s_cv": partial(make_tuev_1s_dataloaders, mode="cv"),

        # BCIC IV 2A Subject Task Within Subjects Classification
        "bciciv_2a_within_classification_motor": partial(make_bciciv_2a_dataloaders, mode="4class_within_subject"),
        "bciciv_2a_within_classification_id": partial(make_bciciv_2a_dataloaders, mode="4class_within_subject", task_type="subject"),

        # Sleep EDF Subject Task Within Subjects Classification
        "sleep_edfx_20subjects_cassette_subjectid": partial(make_sleep_edfx_dataloaders, mode="20subjects_cassette_subjectid", task_type="subject_id"),
        "sleep_edfx_20subjects_cassette_sleep": partial(make_sleep_edfx_dataloaders, mode="20subjects_cassette_sleep"),

        # Kaggle ERN Subject Task Within Subjects Classification
        "kaggle_ern_within_subjects_ern": partial(make_kaggle_ern_dataloaders, mode="within_subjects_ern"),
        "kaggle_ern_within_subjects_subjectid": partial(make_kaggle_ern_dataloaders, mode="within_subjects_subjectid"),

        }


