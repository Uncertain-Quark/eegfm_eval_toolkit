# BCI Competition IV 2a dataset
# https://www.bbci.de/competition/iv/desc_2a.pdf

import os, sys, json, tqdm, mne
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

from biodl.utils.augmentations import *

def download_and_process_dataset(n_folds: int = 5, random_state: int = 42):
    # 1. Setup Paths
    # Using specific subfolders for BCI IV 2a to keep things organized
    data_root = os.path.join(os.getenv("BIODL_RAW_DATA_ROOT", "."), "bciciv_2a_zip")
    preprocessed_data_root = os.path.join(os.getenv("BIODL_PREPROCESSED_DATA_ROOT", "bciciv_2a_preprocessed"))
    split_info_path = os.path.join(os.getenv("BIODL_SPLIT_PATH", "splits"), "bciciv_2a")

    os.makedirs(data_root, exist_ok=True)
    os.makedirs(preprocessed_data_root, exist_ok=True)
    os.makedirs(split_info_path, exist_ok=True)

    # 2. Download the dataset
    # We check if the extraction directory already exists to avoid re-downloading
    extraction_dir = os.path.join(os.getenv("BIODL_RAW_DATA_ROOT", "."), "bciciv_2a")

    if not os.path.exists(extraction_dir):
        print(f"Downloading BCI IV 2a dataset to {data_root}...")

        # URLs
        data_url = "https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip"
        labels_url = "https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip"

        # --- Replicating the Bash Logic in Python ---

        # 1. Download Main GDF Zip
        # wget -P $dest_path ...
        if os.system(f"wget -c -P {data_root} {data_url}") != 0:
            print("Error downloading data zip.")
            return

        # 2. Create Destination & Unzip Data
        # mkdir -p ... && unzip ...
        os.makedirs(extraction_dir, exist_ok=True)
        zip_path = os.path.join(data_root, "BCICIV_2a_gdf.zip")
        os.system(f"unzip -o {zip_path} -d {extraction_dir}")

        # 3. Download Labels
        # wget -P .../BCICIV_2a_labels/ ...
        labels_dest = os.path.join(data_root, "BCICIV_2a_labels")
        os.makedirs(labels_dest, exist_ok=True)
        if os.system(f"wget -c -P {labels_dest} {labels_url}") != 0:
            print("Error downloading labels.")
            return

        # 4. Unzip Labels into the main data folder
        # unzip ... -d .../BCICIV_2a/
        labels_zip_path = os.path.join(labels_dest, "true_labels.zip")
        os.system(f"unzip -o {labels_zip_path} -d {extraction_dir}")

        print("Download and extraction complete.")
    else:
        print(f"Dataset found at {extraction_dir}, skipping download.")


    for i in range(1, 10):

        subjects_train = [j for j in range(1, 10) if j != i]
        subjects_test = [i]

        json.dump(subjects_train, open(os.path.join(split_info_path, f"{(i-1):02}_train.json"), "w"), indent=4)
        json.dump(subjects_test, open(os.path.join(split_info_path, f"{(i-1):02}_test.json"), "w"), indent=4)


class bciciv_2a(Dataset):

    def __init__(self, mode: str='between',
            split_mode: str='train',
            subject_list: list[int]=None):
        # mode: can be between or within subjects
        # split_mode: can be train or eval for between subjects
        # subject_list : if the mode is between, pass the list of subjects to be used for the dataloader
        # Since the dataset collects the data over two days for the same subject, we can define two manners of evaluation i.e. 1. Within subjects by consideting the day one as train and the other as test 2. Between subjects by leaving the data of one subject out and training on the other subjects.

        super().__init__()
        self.sr = 250
        self.n_classes = 4
        self.n_eeg_channels = 22
        # Class 1: Onset left; Class 2: Onset Right; Class 3: Onset Foot; Class 4: Onset Tongue; -100 : Rejected Trials
        self.custom_mapping_events_train = {'769': 1, '770': 2, '771': 3, '772': 4, '1023': -100}
        self.custom_mapping_events_eval = {'768': 0}

        self.data_root = os.path.join(os.getenv("BIODL_RAW_DATA_ROOT", "."), "bciciv_2a")

        if mode == 'within' and split_mode == 'train':
            self.subject_files = [os.path.join(self.data_root, f'A0{i}T.gdf') for i in range(1, 10)]
        elif mode == 'within' and split_mode == 'eval':
            self.subject_files = [os.path.join(self.data_root, f'A0{i}E.gdf') for i in range(1, 10)]
        elif mode == 'between' and subject_list is not None:
            self.subject_files = [os.path.join(self.data_root, f'A0{i}{j}.gdf') for i in subject_list for j in ['T', 'E']]
        else:
            raise ValueError(f'Unexpected Value received for combination of {mode} and {split_mode}')

        # For each of the files identified, process the data
        data_all_subjects, labels_all_subjects, subject_all_list = [], [], []
        for f in tqdm.tqdm(self.subject_files):
            data_subject, label_subject, subject_list = self._process_gdf_file(f)
            data_all_subjects.append(data_subject)
            labels_all_subjects.append(label_subject)
            subject_all_list.extend(subject_list)
        self.data_all_subjects = np.concatenate(data_all_subjects, axis=0, dtype=np.float32)
        self.labels_all_subjects = np.concatenate([l.flatten() for l in labels_all_subjects], axis=0)
        self.subject_all_list = subject_all_list


    def _process_gdf_file(self, file_path):
        subject = int(file_path.split('/')[-1][:3].strip("ATE"))

        raw_gdf = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)

        # Notch Filter at 60Hz
        raw_gdf = raw_gdf.notch_filter(60, method="iir", phase="zero", verbose=False)
        # Bandpass filter at (0.5, 50)Hz
        raw_gdf = raw_gdf.filter(0.5, 50, method="fir", phase="zero", verbose=False)

        non_eog_channels = mne.pick_channels(raw_gdf.ch_names, include=[ch for ch in raw_gdf.ch_names if "eog" not in ch.lower()])
        raw_gdf.pick(non_eog_channels)
        raw_gdf_eeg = raw_gdf.get_data() # 22 channels EEG
        # EEG average re-referencing
        # raw_gdf_eeg = raw_gdf_eeg - np.mean(raw_gdf_eeg, axis=0, keepdims=True)

        # 1. Calculate the mean for each channel (across time)
        channel_means = np.mean(raw_gdf_eeg, axis=1, keepdims=True)

        # 2. Calculate the standard deviation for each channel (across time)
        # The divisor for calculating standard deviation in numpy is typically N by default (population std dev).
        channel_stds = np.std(raw_gdf_eeg, axis=1, keepdims=True)

        # 3. Apply the Z-score formula: (X - mu) / sigma
        # Avoid division by zero: replace zero standard deviations with 1 (or a small epsilon)
        # to prevent NaNs while keeping the mean-subtracted value close to zero.
        # An alternative is to just skip normalization for that channel, but this approach is robust.
        channel_stds[channel_stds == 0] = 1.0

        raw_gdf_eeg = (raw_gdf_eeg - channel_means) / channel_stds

        # If the file is from eval, then it doesn't have the labels in the file
        if 'E' in os.path.basename(file_path):
            # events_gdf has the information about the labels
            events_gdf = mne.events_from_annotations(raw_gdf, self.custom_mapping_events_eval, verbose=False)
            labels = scipy.io.loadmat(file_path[:-3] + 'mat')['classlabel']
            data = [raw_gdf_eeg[:, t + 1*self.sr: t+4*self.sr] for t in events_gdf[0][:, 0]]
            data = np.array(data)
        else:
            # Load the labels from the mat file
            events_gdf = mne.events_from_annotations(raw_gdf, self.custom_mapping_events_train, verbose=False)
            data, labels = [], []
            for i in range(events_gdf[0].shape[0]):
                if events_gdf[0][i, -1] in [1, 2, 3, 4]:
                    check_reject_index = min(i+1, events_gdf[0].shape[0]-1)
                    if events_gdf[0][check_reject_index, -1] != -100:
                        t = events_gdf[0][i, 0]
                        data.append(raw_gdf_eeg[:, t+1*self.sr: t+4*self.sr])
                        labels.append(events_gdf[0][i, -1])
            data, labels = np.array(data), np.array(labels)
        # labels-1 below is to ensure that the class labels lie in the range [0, n_classes-1] rather than between [1, n_classes] which would result in an error.
        return data, labels-1, [subject]*len(data)

    def __len__(self):
        return self.labels_all_subjects.shape[0]

    def __getitem__(self, index):
        return self.data_all_subjects[index], self.labels_all_subjects[index]


CH_NAMES = [
    "Fz", "Fc3", "Fc1", "Fcz", "Fc2", "Fc4",  # Frontal (0-5)
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6", # Central (6-12)
    "CP3", "CP1", "CPz", "CP2", "CP4",        # Centro-Parietal (13-17)
    "P1", "Pz", "P2",                         # Parietal (18-20)
    "POz"                                     # Occipital (21)
]

class bciciv_2a_npz(Dataset):

    def __init__(
            self,
            data_path: str=None,
            channel_idx: list=None,
            label_noise: float=None,
            n_classes: int=4,
            model_type: str=None,
            task_type: str="motor", # motor or subject
            is_adversarial_training: bool=False,
            **kwargs
    ):
        super(bciciv_2a_npz, self).__init__()
        self.data_path = data_path
        self.channel_idx = channel_idx
        self.model_type = model_type
        self.aug_dict = kwargs.get("aug_dict", {})
        self.task_type = task_type
        self.return_metadata = getattr(kwargs, "return_metadata", True)
        self.is_adversarial_training = is_adversarial_training

        self.frontal_indices = [i for i, ch in enumerate(CH_NAMES) if ch.startswith('F')]
        self.peripheral_indices = [
            0, 5,       # Fz, Fc4 (approx)
            6, 12,      # C5, C6
            18, 20, 21  # P1, P2, POz
        ]

        data = np.load(data_path, allow_pickle=True)
        
        self.X = data["X"]
        self.y = data["y"]

        if label_noise is not None and label_noise > 0.0:
            print(f"Applying stratified label noise: {label_noise:.2%}")
            
            # unique classes in the target (assuming 0, 1, 2, 3)
            unique_classes = np.unique(self.y)
            
            for c in unique_classes:
                # 1. Find all indices belonging to THIS class
                # np.where returns a tuple, we take the first element
                indices_of_class = np.where(self.y == c)[0]
                n_class_samples = len(indices_of_class)
                
                # 2. Determine how many to corrupt for THIS class
                n_noisy = int(n_class_samples * label_noise)
                
                # 3. Select random subset of THIS class
                noisy_indices = np.random.choice(indices_of_class, size=n_noisy, replace=False)
                
                # 4. Generate offsets (1 to n_classes-1) to ensure we flip to a DIFFERENT class
                # For 4 classes, offsets are [1, 2, 3]
                noise_offsets = np.random.randint(1, n_classes, size=n_noisy)
                
                # 5. Apply noise
                # Since we know the original label is 'c', we can just do (c + offset) % n_classes
                self.y[noisy_indices] = (self.y[noisy_indices] + noise_offsets) % n_classes
                
            print(f"Noise injection complete.")

        self.keys = data["keys"]

        if self.task_type == "subject":
            self.y = self._extract_subject_labels(self.keys)

        self.key_to_idx = {k: i for i, k in enumerate(self.keys)}
        print(f"Loaded {len(self.X)} trials")

        if self.return_metadata:
            self.metdata = [
                {
                    "subjectid": k.split("_")[0],
                    "class": k.split("_")[2],
                    "session": k.split("_")[1]
                }

                for k in self.keys
            ]

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
        print(f"Unique Subjects: {unique_subjects}")
        sub_to_idx = {sub: i for i, sub in enumerate(unique_subjects)}
        
        # Convert to numpy array of mapped integer targets
        mapped_y = np.array([sub_to_idx[sub] for sub in raw_subject_ids])
        return mapped_y

    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.aug_dict is not None:
            if 'snr_db' in self.aug_dict and self.aug_dict['snr_db'] is not None:
                x = add_gaussian_noise(x, snr_db=self.aug_dict['snr_db'])

            # B. EMG Noise (Peripheral Channels)
            if self.aug_dict.get('emg_noise', False):
                fs = self.aug_dict.get('fs', 250)
                x = add_emg_noise(x, fs, self.peripheral_indices, snr_db=5)

            # C. EOG Noise (Frontal Channels)
            if self.aug_dict.get('eog_noise', False):
                fs = self.aug_dict.get('fs', 250)
                x = add_eog_noise(x, fs, self.frontal_indices, amplitude_factor=3.0)

        # 3. Channel Selection (if subset requested)
        if self.channel_idx is not None:
            # if self.model_type is not None and "cbramod" in self.model_type:
            #     # zero out the non channel_idx instead of selecting channels
            #     temp_arr = np.zeros_like(x)
            #     temp_arr[self.channel_idx] = x[self.channel_idx]
            #     x = temp_arr
            # else:
            x = self.X[idx, self.channel_idx]

        return_dict = {
            "inputs": torch.tensor(x, dtype=torch.float32),
            "labels": torch.tensor(y, dtype=torch.long)
        }
        if self.return_metadata:
            return_dict["metadata"] = self.metdata[idx]

        return return_dict


class bciciv_2a_reconstruction_npz(Dataset):
    def __init__(
            self,
            data_path: str=None,
            channel_idx: list=None,
            model_type: str=None,
            task_type: str="subject",
            **kwargs
    ):
        
        super().__init__()
        self.channel_idx = channel_idx
        self.model_type = model_type
        self.data_path = data_path
        self.aug_dict = kwargs.get("aug_dict", {})
        self.is_adversarial_training = kwargs.get("is_adversarial_training", None)
        self.task_type = task_type

        self.frontal_indices = [i for i, ch in enumerate(CH_NAMES) if ch.startswith('F')]
        self.peripheral_indices = [
            0, 5,       # Fz, Fc4 (approx)
            6, 12,      # C5, C6
            18, 20, 21  # P1, P2, POz
        ]

        data = np.load(data_path, allow_pickle=True)
        
        self.X = data["X"]
        self.y = data["y"]

        self.keys = data["keys"]

        self.key_to_idx = {k: i for i, k in enumerate(self.keys)}
        print(f"Loaded {len(self.X)} trials")

        if self.task_type == "subject":
            self.y = self._extract_subject_labels(self.keys)

    def __len__(self):
        return len(self.X)
    
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
        print(f"Unique Subjects: {unique_subjects}")
        sub_to_idx = {sub: i for i, sub in enumerate(unique_subjects)}
        
        # Convert to numpy array of mapped integer targets
        mapped_y = np.array([sub_to_idx[sub] for sub in raw_subject_ids])
        return mapped_y
    
    def __getitem__(self, idx):

        x = self.X[idx]
        y = self.y[idx]
        
        if self.aug_dict is not None:
            if 'snr_db' in self.aug_dict and self.aug_dict['snr_db'] is not None:
                x = add_gaussian_noise(x, snr_db=self.aug_dict['snr_db'])

            # B. EMG Noise (Peripheral Channels)
            if self.aug_dict.get('emg_noise', False):
                fs = self.aug_dict.get('fs', 250)
                x = add_emg_noise(x, fs, self.peripheral_indices, snr_db=5)

            # C. EOG Noise (Frontal Channels)
            if self.aug_dict.get('eog_noise', False):
                fs = self.aug_dict.get('fs', 250)
                x = add_eog_noise(x, fs, self.frontal_indices, amplitude_factor=3.0)
        
        if self.channel_idx is not None:
            x = x[idx, self.channel_idx]

        if self.is_adversarial_training:
            return {
                "inputs": torch.tensor(x, dtype=torch.float32),
                "labels": torch.tensor(x, dtype=torch.float32),
                "adv_labels": torch.tensor(y, dtype=torch.long)
            }
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)

        




        
