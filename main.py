from eegfm_eval_toolkit.utils.make_dataloaders import DATALOADERS_DICT
from eegfm_eval_toolkit.utils.make_dataset import DATASET_DICT
from eegfm_eval_toolkit.utils.make_models import get_models
from eegfm_eval_toolkit.trainer.trainer import TrainerEEG

from fvcore.nn import FlopCountAnalysis, flop_count_table

import os, sys, json, argparse
from functools import partial

import torch
import numpy as np

class FLOPsWrapper(torch.nn.Module):
    def __init__(self, model, forward_kwargs):
        super().__init__()
        self.model = model
        self.forward_kwargs = forward_kwargs
        
    def forward(self, x):
        # Unpacks the stored arguments into the model's forward pass
        return self.model(x, **self.forward_kwargs)
    
def reproducible(seed: int=42):
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", type=str, help="Directory to store the logs and checkpoints", default=None)
parser.add_argument("--dataset", type=str, help="Dataset key which determines the dataset and specific task")
parser.add_argument("--seed", type=int, default=42, help="Seeds for reproducibility")
parser.add_argument("--percent_train_subjects", type=float, default=None, help="Percentage of subjects to use in training")
parser.add_argument("--percent_train_per_subject", type=float, default=None, help="Percentage of samples per subject")
parser.add_argument("--n_subjects", default=None, type=int, help="Number of subjects to use in training")
parser.add_argument("--n_samples_per_subject", default=None, type=int, help="Number of samples per subject to use in training")
parser.add_argument("--model", type=str, default=None, help="model type to use for training.")
parser.add_argument("--label_noise", type=float, default=None, help="ratio of label noise")
parser.add_argument("--config", default=None, type=str, help="Path to config file to use")
parser.add_argument("--channel_config", default=None, type=str, help="Path to channel config to sample channels from")
parser.add_argument("--batch_size", default=None, type=int, help="batch size to use for training.")
parser.add_argument("--lr", type=float, default=None, help="learning rate to override config")
parser.add_argument("--weight_decay", type=float, default=None, help="weight decay to override config")
parser.add_argument("--n_epochs", type=int, default=None, help="number of epochs to train the model")
parser.add_argument("--eval_batch_size", type=int, default=None, help="batch size for evaluation set")
parser.add_argument("--model_type", type=str, default=None, help="Override model type mentioned in config")
parser.add_argument("--compute_flops", action="store_true", help="Compute FLOPs and params, then exit")
parser.add_argument("--early_stop_patience", type=int, default=None, help="early stop patience value")
parser.add_argument("--lora_r", default=None, type=int)
parser.add_argument("--lora_alpha", default=None, type=int)
parser.add_argument("--lora_dropout", default=None, type=float)
parser.add_argument("--feature", type=str, default=None, help="feature to override config")
parser.add_argument("--n_seconds_input", type=float, default=None, help="input samples length in seconds")
parser.add_argument("--norm_type", type=str, default=None, help="Nomralization type used on the dataset")
args = parser.parse_args()

logdir = args.logdir if args.logdir is not None else os.getenv("eegfm_eval_toolkit_LOGDIR")
dataset = args.dataset
seed = args.seed
model_type = args.model 
channel_config = args.channel_config
batch_size = args.batch_size
model_type = args.model_type
norm_type = args.norm_type

reproducible(seed)

# get the config files
config_path = os.path.join(os.getenv("eegfm_eval_toolkit_ROOT"), "config", "finetune", f"{dataset}.json") if args.config is None else args.config
config = json.load(open(config_path, "r"))

if args.lora_r is not None: config["model_params"]["lora_r"] = args.lora_r
if args.lora_alpha is not None: config["model_params"]["lora_alpha"] = args.lora_alpha
if args.lora_dropout is not None: config["model_params"]["lora_dropout"] = args.lora_dropout
if args.feature is not None: config["feature"] = args.feature
if args.n_seconds_input is not None: config["model_params"]["n_seconds_input"] = args.n_seconds_input

if args.lr is not None: config["trainer_args"]["optimizer_args"]["params"]["lr"] = args.lr
    
if args.weight_decay is not None: config["trainer_args"]["optimizer_args"]["params"]["weight_decay"] = args.weight_decay
if args.n_epochs is not None: 
    config["train_fn_args"]["n_epochs"] = args.n_epochs
    if type(config["trainer_args"]["scheduler_args"]) is dict and config["trainer_args"]["scheduler_args"]["type"] == "cosineAnnealLR":
        config["trainer_args"]["scheduler_args"]["params"]["T_max"] = args.n_epochs
    elif type(config["trainer_args"]["scheduler_args"]) is list:
        for s in config["trainer_args"]["scheduler_args"]:
            if s["type"] == "cosineAnnealLR":
                s["params"]["T_max"] = args.n_epochs
if args.early_stop_patience is not None: config["trainer_args"]["early_stop_patience"] = args.early_stop_patience

# if model_type is not None: config["model_type"] = model_type

if channel_config is not None:
    channel_cfg = json.load(open(channel_config, "r"))
    config["channel_config"] = channel_cfg

if model_type is None:
    model_type = config["model_type"] if "model_type" in config.keys() else "eegnet"
else:
    print(f"Overwriting model type from {config['model_type']} to {model_type}")
    config["model_type"] = model_type
    

args_dataloaders_fn = dict()
if "feature" in config.keys():
    global_dataset_info = DATASET_DICT[dataset](
        feature=config["feature"], fs=config["model_params"]["sampling_rate"],
        model_type=config["model_type"],
        label_noise=args.label_noise
        ) if dataset in DATASET_DICT.keys() else None
else:
    global_dataset_info = DATASET_DICT[dataset](
        fs=config["model_params"]["sampling_rate"],
        model_type=config["model_type"]
    ) if dataset in DATASET_DICT.keys() else None

args_dataloaders_fn = {
    "percent_train_subjects": args.percent_train_subjects,
    "percent_train_per_subject": args.percent_train_per_subject,
    "n_subjects": args.n_subjects,
    "n_samples_per_subject": args.n_samples_per_subject,
    "global_dataset_info": global_dataset_info,
    "channel_config": config.get("channel_config", None),
    "label_noise": args.label_noise,
    "aug_dict": config.get("aug_dict", None),
    "model_type": config.get("model_type", None),
    "seed": args.seed,
    "norm_type": norm_type
}
if args.eval_batch_size is not None: args_dataloaders_fn["eval_batch_size"] = args.eval_batch_size
if "n_seconds_input" in config["model_params"].keys(): args_dataloaders_fn["seq_len"] = config["model_params"]["n_seconds_input"]
if "feature" in config.keys(): args_dataloaders_fn["feature"] = config["feature"]
if "sampling_rate" in config["model_params"].keys(): args_dataloaders_fn["fs"] = config["model_params"]["sampling_rate"]
if batch_size is not None: args_dataloaders_fn["batch_size"] = batch_size


# dataloaders_fn = partial(DATALOADERS_DICT[dataset], percent_train_subjects=args.percent_train_subjects, global_dataset_info=global_dataset_info)
dataloaders_fn = partial(DATALOADERS_DICT[dataset], **args_dataloaders_fn)

def get_logdir_path(logdir_base, dataset, fold, seed, model_type, **kwargs):
    aug_dict = kwargs["aug_dict"]
    # print(kwargs)
    logdir = os.path.join(logdir_base, dataset, model_type, f"fold-{fold}_seed-{seed}")
    # if kwargs["percent_train_per_subject"] is not None and kwargs["percent_train_subjects"] is not None:
    #     logdir += f"_percent_train_subject-{kwargs['percent_train_subjects']}_percent_train_per_subject-{kwargs['percent_train_per_subject']}"
    if kwargs["n_subjects"] is not None:
        logdir += f"_nsubjects-{kwargs['n_subjects']}"
    if kwargs['n_samples_per_subject'] is not None:
        logdir += f"_nsamplespersubject-{kwargs['n_samples_per_subject']}"

    if kwargs["percent_train_subjects"] is not None:
        logdir += f"_percent_train_subject-{kwargs['percent_train_subjects']}"
    if kwargs["percent_train_per_subject"] is not None:
        logdir += f"_percent_train_per_subject-{kwargs['percent_train_per_subject']}"
    
    if kwargs["finetune_type"] is not None:
        logdir += f"_finetune-{kwargs['finetune_type']}"
    
    if kwargs["channel_config"] is not None:
        logdir += f"_channels-{kwargs['channel_config']['name']}"
    
    if kwargs["feature"] is not None:
        logdir += f"_feature-{kwargs['feature']}"
    
    if kwargs["label_noise"] is not None:
        logdir += f"_label-noise-{kwargs['label_noise']}"
    
    if kwargs["aug_dict"] is not None:
        aug_dict_str = "--".join([f"{k}-{v}" for k,v in kwargs["aug_dict"].items()])
        logdir += f"-{aug_dict_str}"
    
    if kwargs["norm_type"] is not None:
        logdir += f"-{kwargs['norm_type']}"

    print(f"Logging Directory: {logdir}")
    return logdir

fold_metrics_list = list()

for fold, channel_idx, channel_names, train_dataloader, val_dataloader, test_dataloader in dataloaders_fn():
    n_channels = len(channel_idx) if channel_idx is not None else None
    channel_names = [channel_names[i] for i in channel_idx] if channel_idx is not None else channel_names

    print(f"fold : {fold}")
    print(f"Length of train: {len(train_dataloader)}")
    print(f"Length of val: {len(val_dataloader)}")
    print(f"Length of test: {len(test_dataloader)}")
    print(f"=======================\n# Channels: {n_channels}\n{channel_names}\n=========================")

    train_dataset = train_dataloader.dataset
    if hasattr(train_dataset, "dataset") and hasattr(train_dataset.dataset, "n_classes"):
        config["model_params"]["n_classes"] = train_dataset.dataset.n_classes
    if n_channels is not None:
        config["model_params"]["n_channels"] = n_channels

    # model definition
    logdir_fold = get_logdir_path(logdir, dataset, fold, seed, model_type, percent_train_subjects=args.percent_train_subjects, 
                                  n_subjects=args.n_subjects, n_samples_per_subject=args.n_samples_per_subject,
                                  percent_train_per_subject=args.percent_train_per_subject, channel_config=config.get("channel_config", None), 
                                  label_noise=args.label_noise, aug_dict=config.get("aug_dict", None), finetune_type=config["model_params"].get("finetune_type", None),
                                  feature=config["feature"], norm_type=norm_type)

    model_info = get_models(config["model_type"], config["model_params"], channel_names=channel_names)
    # model, input_chans = model_info["model"], model_info["input_chans"]
    
    if args.compute_flops:
        print(f"\n[INFO] Computing FLOPs for model: {config['model_type']}")

        n_channels = config["model_params"]["n_channels"]
        model = model_info.pop("model")
        forward_args = {k: v for k, v in model_info.items() if k != "model"}
        
        # 1. Determine Input Shape
        # Extract sampling rate and duration from config to get time points
        sr = config["model_params"].get("sampling_rate", 100) # default to 100 if missing
        duration = config["model_params"].get("n_seconds_input", 1) # default to 1s
        n_timepoints = int(sr * duration)
        
        # 2. Create Dummy Input
        # Shape usually: (Batch=1, Channels, Time)
        # Note: Some EEG models (like EEGNet 2D) might expect (Batch, 1, Channels, Time)
        # If your model expects 4D, uncomment the unsqueeze line below.
        dummy_input = torch.randn(1, n_channels, n_timepoints)
        
        # If model expects 4D input (1, 1, C, T) commonly used in EEGNet implementations:
        # dummy_input = dummy_input.unsqueeze(1) 

        # Ensure model and input are on same device
        device = next(model.parameters()).device
        model.to(device)
        dummy_input = dummy_input.to(device)

        wrapper = FLOPsWrapper(model, forward_args)
        flops = FlopCountAnalysis(wrapper, dummy_input)
        flops.unsupported_ops_warnings(False)

        # 3. Compute and Print
        # flops = FlopCountAnalysis(model, dummy_input)
        print(flop_count_table(flops))
        
        print(f"Total FLOPs: {flops.total() / 1e9:.4f} GFLOPs")
        print("Exiting after FLOP computation.")

        num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
        print(f"Total number of trainable parameters: {num_params/1e6:.3f}M")
        sys.exit(0)

    trainer = TrainerEEG(**model_info, logdir=logdir_fold, **config["trainer_args"], model_type=config["model_type"])
    trainer.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader, test_dataloader=test_dataloader, **config["train_fn_args"])
    fold_metrics = trainer.test_evaluate_classification(test_dataloader)

    json.dump(fold_metrics, open(os.path.join(logdir_fold, "test_results.json"), "w"), indent=4)
    fold_metrics_list.append(fold_metrics)

if "bac" in fold_metrics_list[0]:
    bac_list = [f["bac"] for f in fold_metrics_list]
    print(f"Average BAC: {np.mean(bac_list)} Std BAC: {np.std(bac_list)}")

if "cohen_kappa" in fold_metrics_list[0]:
    kappa_list = [f["cohen_kappa"] for f in fold_metrics_list]
    print(f"Average Cohens Kappa: {np.mean(kappa_list)} Std Kappa: {np.std(kappa_list)}")







