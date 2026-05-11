from eegfm_eval_toolkit.datasets.motor_imagery.mmiphysionet import mmiphysionet_npz
from eegfm_eval_toolkit.datasets.motor_imagery.bciciv_2a import bciciv_2a_npz
from eegfm_eval_toolkit.datasets.normal_abnormal.tuab import tuab

from functools import partial
import os, sys, json, glob

def make_bciciv_2a_dataset(
        feature: str="raw_norm",
        fs: int=250,
        model_type: str="cbramod",
        debug: bool=False,
        **kwargs
):
    
    try:
        data_path = os.path.join(os.getenv("eegfm_eval_toolkit_PREPROCESSED_DATA_ROOT", "./preprocessed"), f"bciciv_2a_{feature}_{fs}", "data.npz")
        dataset = bciciv_2a_npz(data_path=data_path, **kwargs)
    except Exception as e:
        data_path = os.path.join(os.getenv("eegfm_eval_toolkit_PREPROCESSED_DATA_ROOT", "./preprocessed"), f"bciciv_2a_{model_type}_{fs}", "data.npz")
        dataset = bciciv_2a_npz(data_path=data_path, **kwargs)

    return {
        "dataset": dataset
    }

def make_mmiphysionet_dataset(
        feature: str="raw_norm",
        fs: int=160,
        model_type: str="cbramod",
        debug: bool=False,
        **kwargs
):
    try:
        data_path = os.path.join(os.getenv("eegfm_eval_toolkit_PREPROCESSED_DATA_ROOT", "./preprocessed"), f"mmiphysionet_{feature}_{fs}", "data.npz")
        dataset = mmiphysionet_npz(data_path=data_path)
    except Exception as e:
        data_path = os.path.join(os.getenv("eegfm_eval_toolkit_PREPROCESSED_DATA_ROOT", "./preprocessed"), f"mmiphysionet_{model_type}_{fs}", "data.npz")
        dataset = mmiphysionet_npz(data_path=data_path)

    return {
        "dataset": dataset
    }

def make_tuab_dataset(
        feature: str="raw",
        fs: int=200,
        model_type: str="cbramod",
        n_channels: int=21,
        debug: bool=False
):
    return None

DATASET_DICT = {
    # "cbramod_mmiphysionet_4class_cross_subject": make_mmiphysionet_dataset,
    # "mmiphysionet_4class_cross_subject": make_mmiphysionet_dataset,
    # "bciciv_2a_4class_cross_subject": make_bciciv_2a_dataset,
    # "tuab_2class": make_tuab_dataset,
}

NUM_FOLDS_DATASET = {
    "cbramod_mmiphysionet_4class_cross_subject": 5,
    "mmiphysionet_4class_cross_subject": 5,
    "mmiphysionet_109_4class_cbramod": 1,
    "bciciv_2a_4class_subject_specific": 9,
    "bciciv_2a_4class_subject_specific_te": 9,
    "bciciv_2a_4class_within_subject": 1,
    "bciciv_2a_4class_cross_subject": 9,
    "tuab_2class": 1,
    "kaggle_ern_2class": 5,
    "kaggle_ern_within_subjects_ern": 1,
    "kaggle_ern_within_subjects_subjectid": 1,
    "mdd_mal_EC": 5,
    "mdd_mal_EO": 5,
    "bciciv_2a_4class_cross_subject": 9,
    "sleep_edfx_5class_telemetry": 5,
    "errp_hri_loso_cursor": 12,
    "physionetp300_loso": 12,
    "eeg_rtmri_cv_production_voicing": 12,
    "mmiphysionet_autoencoder": 1,
    "mmiphysionet_autoencoder_4class": 1,
    "mmiphysionet_4class_within_subject": 1,
    "mmiphysionet_4class_within_subject_id": 1,
    "mmiphysionet_4class_within_subject_motor": 1,
    "bciciv_2a_4class_within_subject_id": 1,
    "bciciv_2a_4class_within_subject_motor": 1,
    "sleep_edfx_20subjects_cassette_subjectid": 5,
    "sleep_edfx_20subjects_cassette_sleep": 5,
    "bciciv_2a_autoencoder_within_4class": 1,
    "bciciv_2a_autoencoder_within_4class_classification_motor": 1,
    "bciciv_2a_autoencoder_within_4class_classification_id": 1,
    "bciciv_2a_autoencoder_within_4class_classification_motor_all": 1,
    "bciciv_2a_autoencoder_within_4class_classification_id_all": 1,
    "tuab_task": 5,
    "tuab_subject": 5
}