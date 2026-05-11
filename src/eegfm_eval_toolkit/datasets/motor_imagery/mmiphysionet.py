# https://physionet.org/content/eegmmidb/1.0.0/

import os, sys, json, mne, tqdm
import numpy as np
import torch
from torch.utils.data import Dataset

def download_and_process_dataset(n_folds: int=5, random_state: int=42):
    data_root = os.path.join(os.getenv("BIODL_RAW_DATA_ROOT"), "mmi_physionet")
    preprocessed_data_root = os.path.join(os.getenv("BIODL_PREPROCESSED_DATA_ROOT", "mmi_physionet_preprocessed"))
    split_info_path = os.path.join(os.getenv("BIODL_SPLIT_PATH"), "mmi_physionet")

    os.makedirs(preprocessed_data_root, exist_ok=True)
    os.makedirs(split_info_path, exist_ok=True)

    # download the dataset
    if not os.path.exists(data_root):
        os.system(f"aws s3 sync --no-sign-request s3://physionet-open/eegmmidb/1.0.0/ {data_root}")

    # preprocess dataset

    # create the 5 fold splits for train, validation and test subjects
    from sklearn.model_selection import KFold, train_test_split
    ignored_subjects = [88, 90, 92, 100]
    subjects = [i for i in range(1, 110) if i not in ignored_subjects]
    
    kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)

    for  i, (train_index, test_index) in enumerate(kf.split(subjects)):
        subjects_train = [subjects[t] for t in train_index]
        subjects_train, subjects_val = train_test_split(subjects_train, test_size=0.2, random_state=42)
        subjects_test = [subjects[t] for t in test_index]

        json.dump(subjects_train, open(os.path.join(split_info_path, f"{i:02}_train.json"), "w"), indent=4)
        json.dump(subjects_val, open(os.path.join(split_info_path, f"{i:02}_val.json"), "w"), indent=4)
        json.dump(subjects_test, open(os.path.join(split_info_path, f"{i:02}_test.json"), "w"), indent=4)

class mmiphysionet(Dataset):
    n_classes_dict = {"left_right": 2,
            "fist_legs": 2,
            "4_class": 4,
            "real_imagined": 4,
            "within_4class": 4,
            "within_2class": 2}
    rounds_dict = {"left_right": [3, 4, 7, 8, 11, 12],
            "fist_legs": [5, 6, 9, 10, 13, 14],
            "4_class": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "within_4class_train": [3, 4, 5, 6, 7, 8, 9, 10],
            "within_4class_test": [11, 12, 13, 14],
            "within_2class_train": [3, 4, 7, 8],
            "within_2class_test": [11, 12]}
    event_id_dict = {"T1": 0, "T2": 1}
    event_start, event_end = 0, 3

    def __init__(self, mode: str=None, split_mode: str="train", subject_list: list[int]=None,
                 resample_rate: int=160, include_subject_ids_classification: bool=False):
        super().__init__()

        self.data_root = os.path.join(os.getenv("BIODL_RAW_DATA_ROOT", "."), "mmi_physionet")
        self.mode = mode
        self.resample_rate = resample_rate
        self.subject_list = subject_list
        # left_right is betweensubjects left and right fist for both real and imagined
        # fist_legs is between subjects fist and legs classification for both real and imagined
        # 4_class is between subjects for all the classes for both real and imagined
        # within_2class consists within subjects open and close left/right fist
        # within_4class consists within subjects all 4 classes
        # real_imagined is within subjects 2 or 4 class classification by taking training data as the real events and testing on imagined trials as the test events

        # Subjects 88, 90, 92 and 100 are to be ignored
        if mode == "real_imagined" or mode == "within" and subject_list is None:
            self.subject_folders = [os.path.join(self.data_root, f"S{i:03d}") for i in range(1, 110) if i not in [88, 90, 92, 100]]
        elif subject_list is not None:
            self.subject_folders = [os.path.join(self.data_root, f"S{i:03d}") for i in subject_list if i not in [88, 90, 92, 100]]

        self.n_classes = mmiphysionet.n_classes_dict[mode]
        self.rounds = mmiphysionet.rounds_dict[mode if not ("within" in mode) else f"{mode}_{split_mode}"]

        # make the data samples
        self._make_data()

        if include_subject_ids_classification:
            self._make_labels_include_subject_ids()

    def _make_labels_include_subject_ids(self):
        labels = [f"{sub_id}_{label}" for sub_id, label in zip(self.subjects_trials_files, self.y)]
        uniq_labels = list(set(labels))
        uniq_labels.sort()

        labels_map_dict = {uniq_label: i for i, uniq_label in enumerate(uniq_labels)}
        labels = [labels_map_dict[l] for l in labels]
        self.labels_map_dict = labels_map_dict
        self.y = labels

        print(f"Number of unique labels: {len(uniq_labels)} {uniq_labels}")
        # import pdb; pdb.set_trace()
        self.n_classes = len(uniq_labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def _make_data(self):
        files = [os.path.join(self.data_root, f"S{subject:03d}", f"S{subject:03d}R{r:02d}.edf") for subject in self.subject_list for r in self.rounds]

        trials_files, labels_files, subjects_trials_files = [], [], []
        # For each of the files, generate the data points and labels
        for edf_file in tqdm.tqdm(files):
            subject = os.path.basename(edf_file).split("R")[0]
            trials_file, labels_file = self._process_edf_file(edf_file)
            # If the mode is 4 class, then change the labels for both fists and both feet to 2,3 instead of 0,1
            round = int(edf_file.split("/")[-1].split("R")[-1].strip(".edf"))
            if (self.n_classes == 4) and (round in [5, 6, 9, 10, 13, 14]):
                labels_file += 2

            # Average re-referencing EEG
            trials_file = trials_file - np.mean(trials_file, axis=1, keepdims=True)

            trials_files.append(trials_file)
            labels_files.append(labels_file)
            subjects_trials_files.extend([subject]*trials_file.shape[0])
        self.X, self.y = np.concatenate(trials_files, axis=0), np.concatenate(labels_files, axis=0)
        self.labels_trials = np.concatenate(labels_files, axis=0)
        self.subjects_trials_files = subjects_trials_files

        print(f"Shape of the input: {self.X.shape} Shape of labels: {self.y.shape}")

    def _process_edf_file(self, edf_file):
        trials, labels = [], []
        raw_edf = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        events, events_dict = mne.events_from_annotations(raw_edf, event_id = mmiphysionet.event_id_dict, verbose=False)

        # Powerline noise removal
        raw_edf = raw_edf.notch_filter(60, method="iir", phase="zero", verbose=False)
        # Bandpass filtering
        raw_edf = raw_edf.filter(0.5, 50, method="fir", phase="zero", verbose=False)

        # Resampling the signal to the desired frequency
        if int(raw_edf.info["sfreq"]) == self.resample_rate:
            raw_edf_resample, events_resample = raw_edf.copy(), events.copy()
        else:
            raw_edf_resample, events_resample = raw_edf.resample(self.resample_rate, events=events[0], method="polyphase")
        # EEG Data without the MNE python wrapper
        raw_data = raw_edf_resample.get_data()

        # Z-score normalization
        channel_means = np.mean(raw_data, axis=1, keepdims=True)

        # 2. Calculate the standard deviation for each channel (across time)
        # The divisor for calculating standard deviation in numpy is typically N by default (population std dev).
        channel_stds = np.std(raw_data, axis=1, keepdims=True)

        # 3. Apply the Z-score formula: (X - mu) / sigma
        # Avoid division by zero: replace zero standard deviations with 1 (or a small epsilon)
        # to prevent NaNs while keeping the mean-subtracted value close to zero.
        # An alternative is to just skip normalization for that channel, but this approach is robust.
        channel_stds[channel_stds == 0] = 1.0

        raw_data = (raw_data - channel_means) / channel_stds

        for event in events_resample:
            start, label = event[0], event[2]
            start_index, end_index = start + int(mmiphysionet.event_start * self.resample_rate), start + int(mmiphysionet.event_end * self.resample_rate)
            trial = raw_data[:, start_index:end_index]

            trials.append(trial)
            labels.append(label)

        return np.array(trials, dtype=np.float32), np.array(labels, dtype=np.int64)

class mmiphysionet_npz(Dataset):
    def __init__(
            self,
            data_path: str=None,
            channel_idx: list=None,
            model_type: str=None,
            task_type: str="motor", # motor or subject
            **kwargs
    ):
        super(mmiphysionet_npz, self).__init__()
        self.data_path = data_path
        self.channel_idx = channel_idx
        self.model_type = model_type
        self.task_type = task_type

        self.return_metadata = getattr(kwargs, "return_metadata", True)

        data = np.load(data_path, allow_pickle=True)
        
        self.X = data["X"]
        self.y = data["y"]
        self.keys = data["keys"]

        self.eegfm_channel_idx = None
        if "eegfm_channel_idx" in data:
            self.eegfm_channel_idx = data["eegfm_channel_idx"]

        if self.task_type == "subject":
            self.y = self._extract_subject_labels(self.keys)

        self.key_to_idx = {k: i for i, k in enumerate(self.keys)}

        if self.return_metadata:
            self.metdata = [
                {
                    "subjectid": k.split("_")[0],
                    "class": k.split("_")[1]
                }

                for k in self.keys
            ]
        print(f"Loaded {len(self.X)} trials.")

    def _extract_subject_labels(self, keys):
        """
        Extracts subject IDs from the keys and maps them to 0-indexed integers.
        Key Format: S001_C0_03_001 -> Subject 1
        """
        # Extract raw integers (e.g., 1, 2, 3, 4)
        raw_subject_ids = [int(k.split('_')[0][1:]) for k in keys]
        
        # PyTorch expects labels to start at 0 and go up to num_classes-1.
        # So we map the raw subject IDs to 0, 1, 2, etc.
        unique_subjects = sorted(list(set(raw_subject_ids)))
        sub_to_idx = {sub: i for i, sub in enumerate(unique_subjects)}
        
        # Convert to numpy array of mapped integer targets
        mapped_y = np.array([sub_to_idx[sub] for sub in raw_subject_ids])
        return mapped_y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.channel_idx is None:
            x = self.X[idx]
        # elif self.model_type is not None and "cbramod" in self.model_type:
        #     # zero out the non channel_idx instead of selecting channels
        #     x = np.zeros_like(self.X[idx])
        #     x[self.channel_idx] = self.X[idx, self.channel_idx]
        else:
            x = self.X[idx, self.channel_idx]
        
        y = self.y[idx]

        return_dict = {
            "inputs": torch.tensor(x, dtype=torch.float32),
            "labels": torch.tensor(y, dtype=torch.long)
        }
        if self.return_metadata:
            return_dict["metadata"] = self.metdata[idx]

        if "eegfm" in self.model_type.lower() and self.eegfm_channel_idx is not None:
            idx_tensor = torch.tensor(self.eegfm_channel_idx, dtype=torch.long)
            return_dict["channel_ids"] = idx_tensor
        return return_dict

class mmiphysionet_reconstruction_npz(Dataset):
    def __init__(
            self,
            data_path: str=None,
            channel_idx: list=None,
            model_type: str=None
    ):
        super(mmiphysionet_reconstruction_npz, self).__init__()
        self.data_path = data_path
        self.channel_idx = channel_idx
        self.model_type = model_type

        data = np.load(data_path, allow_pickle=True)
        
        self.X = data["X"]
        self.y = data["y"]
        self.keys = data["keys"]

        self.key_to_idx = {k: i for i, k in enumerate(self.keys)}
        print(f"Loaded {len(self.X)} trials.")

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.channel_idx is None:
            x = self.X[idx]
        else:
            x = self.X[idx, self.channel_idx]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)


if __name__ == "__main__":
    download_and_process_dataset()




