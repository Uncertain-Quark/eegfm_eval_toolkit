import os, sys, json
import pandas as pd 

class BaseDataset:
    def __init__(
            self,
            dataset_name: str=None,
            n_folds: int=None,
    ):
        self.dataset_name = dataset_name
        self.n_folds = n_folds

        root = os.getenv("BIODL_RAW_DATA_ROOT", "./data")
        self.data_root = os.path.join(root, dataset_name)

        self.pre_root = os.getenv("BIODL_PREPROCESSED_DATA_ROOT", "./data_preprocessed")
        self.split_path = os.path.join(os.getenv("BIODL_SPLIT_PATH", "./splits"), dataset_name)

        os.makedirs(self.split_path, exist_ok=True)
    
    def download_dataset():
        raise NotImplementedError
    
    def process_single_file(filepath: str=None, config: dict=None):
        raise NotImplementedError
    
    def build_lmdb(configs: list=None):
        raise NotImplementedError
    
    def generate_splits(df):
        raise NotImplementedError