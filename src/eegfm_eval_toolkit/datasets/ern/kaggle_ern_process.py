# Kaggle ERN dataset
# Perrin, M., Maby, E., Daligault, S., Bertrand, O., & Mattout, J. Objective and subjective evaluation of online error correction during P300-based spelling. Advances in Human-Computer Interaction, 2012, 4. (link)

# dataset can be used to test the following:
# generalization of ERN tasks between subjects
# personalization through training on first four sessions and using session 5 as testing data

import os, sys, json, pickle
import numpy as np
import pandas as pd 
import mne 
import lmdb
import tqdm

from glob import glob
from sklearn.model_selection import train_test_split, KFold

from biodl.utils.make_ssl_data import process_ssl_data

# Dataset Configuration
dataset_name = "kaggle_ern"
SAMPLING_RATE = 200
RANDOM_STATE = 42
EPOCH_LENGTH = 1

NON_EEG_COLUMNS = ["Time", "EOG"]
ANNOTATION_COLUMN = "FeedBackEvent"


def download_dataset(data_root):
    if os.path.exists(data_root):
        print(f"    Dataset Already downloaded. Skipping dataset download!")
    else:
        # download dataset zip file
        os.system(f"kaggle competitions download -c inria-bci-challenge -p {data_root}")

        # download test dataset true labels
        os.system(f"wget https://storage.googleapis.com/kaggle-forum-message-attachments/80787/2570/true_labels.csv -P {data_root}")
        
        # unzip downloaded dataset zip file to data_root
        os.system(f"unzip {data_root}/inria-bci-challenge.zip -d {data_root}")

        # unzip train and test data
        os.system(f"unzip {data_root}/test.zip -d {data_root}/test/")
        os.system(f"unzip {data_root}/train.zip -d {data_root}/train/")

        print(f"Downloaded dataset: {dataset_name} to {data_root}")

def get_paths():
    data_root = os.path.join(os.getenv("BIODL_RAW_DATA_ROOT", "./"), dataset_name)

    pre_root = os.path.join(os.getenv("BIODL_PREPROCESSED_DATA_ROOT", "./data_preprocessed"))
    split_path = os.path.join(os.getenv("BIODL_SPLIT_PATH", "./splits"), dataset_name)
    os.makedirs(split_path, exist_ok=True)

    return data_root, pre_root, split_path

def build_lmdb(data_root, pre_root, configs):
    
    
    # 1. Load Labels
    train_labels_df = pd.read_csv(os.path.join(data_root, "TrainLabels.csv"))
    test_labels_df = pd.read_csv(os.path.join(data_root, "true_labels.csv"), header=None, names=['Prediction'])
    
    # 2. Get Files (Sorted to match true_labels logic)
    train_files = sorted(glob(os.path.join(data_root, "train", "Data*_Sess*.csv")))
    test_files = sorted(glob(os.path.join(data_root, "test", "Data*_Sess*.csv")))
    
    dfs = {}
    for config in configs:
        lmdb_path = os.path.join(pre_root, f"{dataset_name}_{config['name']}")
        os.makedirs(lmdb_path, exist_ok=True)

        env = lmdb.open(lmdb_path, map_size=1099511627776) # 1TB
        meta_data = []

        def process_set(files, label_source, split_type):
            label_idx = 0
            channels = None
            for fpath in tqdm.tqdm(files, desc=f"Processing {split_type}"):
                fname = os.path.basename(fpath).replace(".csv", "")
                data_dict = process_single_file(fpath, config)
                
                if data_dict is None: continue
                
                if channels is None: channels=data_dict["channels"]
                # Extract epochs and match with labels
                # Every time FeedBackEvent jumps from 0 to 1, a new trial starts
                epochs = data_dict['epochs']
                
                with env.begin(write=True) as txn:
                    for i, epoch in enumerate(epochs):
                        # Unique key: Subject_Sess_File_Trial
                        key_str = f"{fname.replace('Data_','')}_T{i:03d}"
                        
                        # Get label from external source
                        # For train, we match the ID; for test, we assume order
                        if split_type == 'train':
                            # Find label where IdFeedBack starts with fname
                            # Note: TrainLabels.csv has IDs like S02_Sess01_FB001
                            # Our fname is S02_Sess01_Data
                            row_id = f"{fname.replace('Data_','')}_FB{i+1:03d}"
                            label = int(train_labels_df.loc[train_labels_df['IdFeedBack'] == row_id, 'Prediction'].values[0])
                        else:
                            label = int(label_source.iloc[label_idx]['Prediction'])
                            label_idx += 1
                        
                        txn.put(key_str.encode('ascii'), pickle.dumps(epoch))
                        meta_data.append({
                            "key": key_str,
                            "split_source": split_type,
                            "label": label,
                            "samples": epoch.shape[1]
                        })
            return channels

        channels_train = process_set(train_files, train_labels_df, 'train')
        channels_test = process_set(test_files, test_labels_df, 'test')
        
        # write the channels list
        channels_path = os.path.join(lmdb_path, "channels.json")
        json.dump(channels_train, open(channels_path, "w"), indent=4)
        
        env.close()
        df = pd.DataFrame(meta_data)
        # import pdb;pdb.set_trace()
        df.to_csv(os.path.join(pre_root, f"{dataset_name}_{config['name']}_metadata.csv"), index=False)
        dfs[config['name']] = df
    return dfs

def process_single_file(file_path, config):
    try:
        eeg_df = pd.read_csv(file_path)
        annots = eeg_df[ANNOTATION_COLUMN].values
        eeg_df = eeg_df.drop(NON_EEG_COLUMNS + [ANNOTATION_COLUMN], axis=1)

        eeg_channels = list(eeg_df.columns)
        mne_info = mne.create_info(ch_names=eeg_channels, ch_types="eeg", sfreq=SAMPLING_RATE)
        raw = mne.io.RawArray(eeg_df.values.T * 1e-6, mne_info, verbose=False)
        raw.set_eeg_reference(ref_channels='average')

        if len(raw.info['bads']) > 0:
                print('interpolate_bads')
                raw.interpolate_bads()

        # Preprocessing
        raw.notch_filter(np.arange(50, config["fs"]/2, 50), verbose=False)
        raw.filter(l_freq=0.1, h_freq=40, verbose=False) # standard ERN filter
        if config["fs"] != SAMPLING_RATE:
            raw.resample(config["fs"], verbose=False, n_jobs=5, method="polyphase")

        data = raw.get_data(units="uV").astype(np.float32)

        if cfg.get('ssl', None) is not None:
            # print(f"Entered SSL loop: {cfg.get('ssl')}")
            data, eeg_channels = process_ssl_data(data, eeg_channels, ssl=cfg['ssl'])
            # import pdb; pdb.set_trace()
        # Z-score
        if config["norm"]:
            data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-6)

        # Epoching: extract 1.3 seconds after feedback start (standard for ERN)
        # Find indices where FeedbackEvent switches from 0 to 1
        diff = np.diff(annots, prepend=0)
        event_indices = np.where(diff == 1)[0]
        
        epochs = []
        epoch_len = int(EPOCH_LENGTH * config["fs"]) # 1.3s window
        
        for idx in event_indices:
            if idx + epoch_len <= data.shape[1]:
                epochs.append(data[:, idx : idx + epoch_len])
        
        return {"epochs": epochs, "channels": eeg_channels}
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def generate_within_subject_splits(df, split_path):
    print("\n--- Generating 5-Fold Within-Subject Splits (ERN & Subject ID) ---")
    
    # Isolate only the 'train' pool from Kaggle, as the 'test' pool doesn't 
    # expose the subject/session mapping in the filenames directly.
    # train_pool = df[df['split_source'] == 'train'].copy()
    train_pool = df
    
    # Extract Subject and Session from the key (e.g., 'S02_Sess01_T000')
    train_pool['subj'] = train_pool['key'].apply(lambda x: x.split('_')[0])
    train_pool['sess'] = train_pool['key'].apply(lambda x: x.split('_')[1])
    
    unique_subjects = np.sort(train_pool['subj'].unique())
    
    subj_mapping = {subj: idx for idx, subj in enumerate(unique_subjects)}
    
    train_val_df = train_pool[train_pool["sess"].isin(["Sess01", "Sess02", "Sess03", "Sess04"])]
    test_df = train_pool[train_pool["sess"].isin(["Sess05"])]

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_val_df["subj"]
    )
        
    data_ern = {"train": [], "val": [], "test": []}
    data_subjid = {"train": [], "val": [], "test": []}
    
    # Populate the JSON structures
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for _, row in split_df.iterrows():
            # Shared base metadata
            base_item = {
                "key": row['key'], 
                "length": int(row['samples']),
                "subject": row['subj']
            }
            
            # ERN Task: use the original feedback label
            ern_item = base_item.copy()
            ern_item["label"] = int(row['label'])
            data_ern[split_name].append(ern_item)
            
            # Subject ID Task: use the mapped 0-3 subject label
            subjid_item = base_item.copy()
            subjid_item["label"] = subj_mapping[row['subj']]
            data_subjid[split_name].append(subjid_item)
            
    # Save Fold-specific JSONs
    ern_out_path = os.path.join(split_path, f"within_subjects_ern_fold0.json")
    with open(ern_out_path, 'w') as f:
        json.dump(data_ern, f, indent=4)
        
    subjid_out_path = os.path.join(split_path, f"within_subjects_subjectid_fold0.json")
    with open(subjid_out_path, 'w') as f:
        json.dump(data_subjid, f, indent=4)
        
    print(f"Fold 0 processed for Subjects: {unique_subjects}")

def generate_splits(df, split_path):
    print("\n--- Generating 5-Fold Subject-Independent Splits ---")
    
    # 1. Isolate the training pool and test set
    train_pool = df[df['split_source'] == 'train'].copy()
    test_df = df[df['split_source'] == 'test']
    
    # 2. Extract unique Subject IDs from the keys (e.g., 'S02' from 'S02_Sess01...')
    train_pool['subj'] = train_pool['key'].apply(lambda x: x.split('_')[0])
    test_df['subj'] = test_df['key'].apply(lambda x: x.split('_')[0])
    unique_subjects = np.sort(train_pool['subj'].unique())

    # 3. Initialize KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # 4. Iterate through folds
    for fold_idx, (train_subj_idx, val_subj_idx) in enumerate(kf.split(unique_subjects)):
        train_subjs = unique_subjects[train_subj_idx]
        val_subjs = unique_subjects[val_subj_idx]
        
        # Create dataframes for this specific fold
        fold_train_df = train_pool[train_pool['subj'].isin(train_subjs)]
        fold_val_df = train_pool[train_pool['subj'].isin(val_subjs)]
        
        data = {
            "train": [],
            "val": [],
            "test": []
        }
        
        # Populate the JSON structure
        for name, split_df in [("train", fold_train_df), ("val", fold_val_df), ("test", test_df)]:
            data[name] = [
                {
                    "key": row['key'], 
                    "label": int(row['label']), 
                    "length": int(row['samples']),
                    "subject": row['subj'] if 'subj' in row else "test_subject"
                }
                for _, row in split_df.iterrows()
            ]
            
        # 5. Save fold-specific JSON
        out_path = os.path.join(split_path, f"2class_fold{fold_idx}.json")
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"Fold {fold_idx}: {len(train_subjs)} Train Subjs, {len(val_subjs)} Val Subjs ({val_subjs})")

if __name__ == "__main__":
    data_root, pre_root, split_path = get_paths()
    download_dataset(data_root)

    configs = [
        {'name': 'raw_norm_200', 'fs': 200, 'norm': True},
        # {'name': 'csbrain_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'csbrain_200'},
        # {'name': 'labram_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'labram_200'},
        # {'name': 'cbramod_200', 'fs': 200, 'norm': False, 'spec': False, 'ssl': 'cbramod_200'}
    ]

    for cfg in configs:
        meta_df_path = os.path.join(os.getenv("BIODL_PREPROCESSED_DATA_ROOT"), f"{dataset_name}_{cfg['name']}_metadata.csv")

        meta_df = build_lmdb(data_root, pre_root, [cfg])
        meta_df = meta_df[cfg['name']]

        if meta_df is not None:
            generate_splits(meta_df, split_path)
            generate_within_subject_splits(meta_df, split_path)




