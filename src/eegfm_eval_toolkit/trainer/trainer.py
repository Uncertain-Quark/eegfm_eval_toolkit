
import os, shutil, tempfile, tqdm
import time 
from typing import Union

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast

import numpy as np

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

# sklearn metrics
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, top_k_accuracy_score, cohen_kappa_score, roc_auc_score

# Ray tune imports
import ray
import ray.tune

from functools import partial

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

def calculate_eer_binary(y_true, y_score):
    """
    Helper function to calculate EER for a single binary task 
    using interpolation and root finding.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    
    # specialized function for the root finder
    # We look for x where (1 - x) - tpr(x) = 0  =>  (1 - tpr) = x  => FNR = FPR
    def eer_root(x):
        return 1. - x - interp1d(fpr, tpr)(x)

    # brentq finds the zero of the function in the range [0, 1]
    eer = brentq(eer_root, 0., 1.)
    return eer

def compute_eer(y_true, y_score):
    """
    Computes EER valid for both Binary and Multi-class classification.
    For multi-class, calculates the macro-average of One-vs-Rest EERs.
    """
    # Convert tensors to numpy if necessary
    if hasattr(y_true, 'cpu'): y_true = y_true.cpu().numpy()
    if hasattr(y_score, 'cpu'): y_score = y_score.cpu().numpy()

    # Case 1: Multi-class (Shape: N x Classes, where Classes > 2)
    if y_score.ndim > 1 and y_score.shape[1] > 2:
        n_classes = y_score.shape[1]
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        
        eer_list = []
        for i in range(n_classes):
            # Treat class i as "1" and all others as "0"
            eer_val = calculate_eer_binary(y_true_bin[:, i], y_score[:, i])
            eer_list.append(eer_val)
        
        return np.mean(eer_list)

    # Case 2: Binary Classification
    else:
        # If y_score is (N, 2), take the positive class column
        if y_score.ndim > 1 and y_score.shape[1] == 2:
            y_score = y_score[:, 1]
            
        return calculate_eer_binary(y_true, y_score)

METRICS = {
    "f1_macro": partial(f1_score, average="macro"),
    "bac": balanced_accuracy_score,
    "acc": accuracy_score,
    "topk_acc": top_k_accuracy_score,
    "cohen_kappa": cohen_kappa_score,
    "auroc": partial(roc_auc_score, multi_class='ovr'),
    "eer": compute_eer 
}

class TrainerEEG:
    def __init__(
            self,
            model=None,
            model_checkpoint: str=None,
            logdir: str=None,
            device: str="cuda",
            optimizer_args=None,
            scheduler_args=None,
            save_n_iters=None,
            save_n_epochs=None,
            patience_metric="bac",
            early_stop_patience=3,
            metrics: Union[list, str]=None,
            gradient_clipping: float=None,
            input_chans: list=None,
            model_type: str=None,
            **kwargs
            ):
        
        self.input_chans = input_chans # used in LaBram

        self.model = model.to(device)
        self.device = device
        self.metrics = metrics if metrics else []
        self.gradient_clipping = gradient_clipping
        self.model_type = model_type
        
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)

        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir=logdir)

        self.history = {
            "train_loss_iter": [], 
            "train_iters": [],
            "train_loss_epoch": [],
            "eval_loss": [], 
            "eval_iters": []
        }
        
        self.running_train_loss = 0.0
        self.running_train_steps = 0
        
        self.test_loss, self.test_metrics = 0, dict()
        self.train_metrics, self.eval_metrics = dict(), dict()

        # Metric parsing
        if isinstance(self.metrics, str):
            self.metrics = [self.metrics]
        
        for m in self.metrics:
            self.eval_metrics[m] = list()

        self.n_iters, self.n_epoch = 0, 0
        self.save_n_iters, self.save_n_epochs = save_n_iters, save_n_epochs

        params_dict = optimizer_args["params"].copy()

        if "labram" in self.model_type:
            # labram has layer wise lr decay
            lr = params_dict.pop("lr", 1e-4)
            weight_decay = params_dict.pop("weight_decay", 0.0)
            optim_groups = self.model.get_optimizer_params(weight_decay=weight_decay, lr=lr)
            for i, group in enumerate(optim_groups):
                print(f"Group {i}: LR={group['lr']:.2e}, WD={group['weight_decay']:.2e}, Params={len(group['params'])}")
        else:
            weight_decay = params_dict.pop("weight_decay", 0.0)
            optim_groups = self.model.get_optimizer_params(weight_decay=weight_decay)

        # optim_groups = self.model.get_optimizer_params(weight_decay=weight_decay)
        # Optimizer Setup
        if optimizer_args["type"] == "adam":
            self.optimizer = torch.optim.Adam(optim_groups, **params_dict)
        elif optimizer_args["type"] == "adamw":
            self.optimizer = torch.optim.AdamW(optim_groups, **params_dict)

        self.scheduler_args = scheduler_args
        self.scheduler = self._get_scheduler(scheduler_args) if scheduler_args else None

        if model_checkpoint is not None:
            self._load_model(model_checkpoint)

        self.early_stop_counter = 0
        self.early_stop_patience = early_stop_patience
        self.patience_metric = "loss" if patience_metric is None else patience_metric
        self.best_checkpoint_path = os.path.join(logdir, "best_model.pt")
        self.best_val = -float('inf') if self.patience_metric != "loss" else float('inf')

    def _get_scheduler_type(self, scheduler_args: dict=None):
        if scheduler_args["type"] == "expLR":
            return ExponentialLR(self.optimizer, **scheduler_args["params"])
        elif scheduler_args["type"] == "linearLR":
            return LinearLR(self.optimizer, **scheduler_args["params"])
        elif scheduler_args["type"] == "cosineAnnealLR":
            return CosineAnnealingLR(self.optimizer, **scheduler_args["params"])
        elif scheduler_args["type"] == "reducelronplateau":
            return ReduceLROnPlateau(self.optimizer, **scheduler_args["params"])

    def _get_scheduler(self, scheduler_args: Union[list, dict] = None):
        if scheduler_args is None: return None
        if isinstance(scheduler_args, dict):
            return self._get_scheduler_type(scheduler_args)
        elif isinstance(scheduler_args, list):
            list_schedulers, milestones, current_milestone = [], [], 0
            for i, s_args in enumerate(scheduler_args):
                if s_args["type"] == "reducelronplateau": raise ValueError("ReduceLROnPlateau cannot be in SequentialLR chain.")
                list_schedulers.append(self._get_scheduler_type(s_args))
                if i < len(scheduler_args) - 1:
                    current_milestone += s_args["params"]["total_iters"]
                    milestones.append(current_milestone)
            return SequentialLR(self.optimizer, schedulers=list_schedulers, milestones=milestones)
        raise NotImplementedError

    def get_iterator(self, start_epoch, n_epochs):
        return tqdm.tqdm(range(start_epoch, n_epochs), desc="Epochs")
    
    def train(self, n_epochs, n_iters=None, train_dataloader=None, val_dataloader=None, test_dataloader=None):
        if n_epochs is None and n_iters is not None:
            n_epochs = int(n_iters // len(train_dataloader))
        
        # Default save frequency if not provided
        if self.save_n_iters is None:
            if self.save_n_epochs is not None:
                self.save_n_iters = self.save_n_epochs * len(train_dataloader)
            else:
                self.save_n_iters = len(train_dataloader) # Default to 1 epoch

        start_epoch = self.n_epoch
        iterator = self.get_iterator(start_epoch, n_epochs)

        for n in iterator:
            epoch_loss = self._train_one_epoch(train_dataloader, val_dataloader)
            
            # Log Epoch Loss
            self.history["train_loss_epoch"].append(epoch_loss)
            self.writer.add_scalar("Loss/Train_Epoch", epoch_loss, n)
            
            # Update progress bar
            # iterator.set_postfix({"Epoch Loss": f"{epoch_loss:.4f}"})
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"Epoch Loss": f"{epoch_loss:.4f}"})

            self.n_epoch = n + 1

            if hasattr(self, "early_stop_counter") and self.early_stop_patience and self.early_stop_counter >= self.early_stop_patience:
                print(f"Early stopping triggered at epoch {n}")
                break

        return self.best_val

    def _train_one_epoch(self, train_dataloader, val_dataloader):
        self.model.train()
        
        epoch_total_loss = 0.0
        epoch_total_samples = 0

        for i, batch in enumerate(train_dataloader):
            adv_labels=None
            if type(batch) == dict:
                inputs = batch.pop("inputs")
                labels = batch.pop("labels", None)
                channel_ids = batch.pop("channel_ids", None)
                adv_labels = batch.pop("adv_labels", None)
            else:
                inputs, labels = batch
                channel_ids = None

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            forward_kwargs = {"labels": labels}
            if channel_ids is not None:
                # Only add these if the dataset provided them (your new model)
                channel_ids = channel_ids.to(self.device)
                forward_kwargs["channel_ids"] = channel_ids
                forward_kwargs["padding_mask"] = None
            if adv_labels is not None:
                adv_labels = adv_labels.to(self.device)
                forward_kwargs["adv_labels"] = adv_labels

            self.optimizer.zero_grad()
            outputs = self.model(inputs, input_chans=self.input_chans, **forward_kwargs) 
            loss = outputs["loss"]
            
            loss.backward()

            if self.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clipping)
            self.optimizer.step()

            loss_val = loss.item()
            batch_size = inputs.size(0)

            self.running_train_loss += loss_val
            self.running_train_steps += 1

            epoch_total_loss += loss_val
            epoch_total_samples += 1

            # Log Iteration Loss
            if self.n_iters % self.save_n_iters == 0 and self.n_iters > 0:
                self._evaluate_classification(val_dataloader)
                
                # Scheduler Step (Plateau)
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(self.history["eval_loss"][-1])

                self.model.train()
                
                avg_iter_loss = self.running_train_loss / self.running_train_steps
                
                self.history["train_loss_iter"].append(avg_iter_loss)
                self.history["train_iters"].append(self.n_iters)
                self.writer.add_scalar("Loss/Train_Iter", avg_iter_loss, self.n_iters)
                
                # Reset Running Counters
                self.running_train_loss = 0.0
                self.running_train_steps = 0
                self.writer.flush()

            self.n_iters += 1

        if not isinstance(self.scheduler, ReduceLROnPlateau) and self.scheduler is not None:
            self.scheduler.step()
            
        # Return the average loss for this epoch
        return epoch_total_loss / epoch_total_samples

    def _save_model(self, checkpoint_path: str=None):
        torch.save({
            "epoch": self.n_epoch, 
            "n_iters": self.n_iters, 
            "history": self.history, # Saved the unified history dict
            "eval_metrics": self.eval_metrics,
            "model_state_dict": self.model.state_dict(), 
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else "", 
            "test_metrics": self.test_metrics
        }, checkpoint_path)

    def _load_model(self, checkpoint_path: str=None):
        info = torch.load(checkpoint_path)
        self.n_epoch = info["epoch"]
        self.n_iters = info["n_iters"]
        
        if "history" in info:
            self.history = info["history"]
        else:
            self.history["train_loss_iter"] = info.get("train_loss", {}).get("loss", [])
            self.history["eval_loss"] = info.get("eval_loss", {}).get("loss", [])

        self.eval_metrics = info["eval_metrics"]
        self.test_metrics = info["test_metrics"]
        self.model.load_state_dict(info["model_state_dict"])
        self.optimizer.load_state_dict(info["optimizer_state_dict"])
        if self.scheduler: self.scheduler.load_state_dict(info["scheduler_state_dict"])

    @torch.no_grad()
    def _evaluate_classification(self, dataloader, split="val"):
        self.model.eval()
        eval_loss, total_samples = 0.0, 0
        preds, gts = [], []
        all_probs = []

        for batch in dataloader:
            adv_labels = None
            # inputs, labels = batch
            if type(batch) == dict:
                inputs = batch.pop("inputs")
                labels = batch.pop("labels")
                channel_ids = batch.pop("channel_ids", None)
                adv_labels = batch.pop("adv_labels", None)
            else:
                inputs, labels = batch
                channel_ids = None

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            forward_kwargs = {"labels": labels}
            if channel_ids is not None:
                # Only add these if the dataset provided them (your new model)
                channel_ids = channel_ids.to(self.device)
                forward_kwargs["channel_ids"] = channel_ids
                forward_kwargs["padding_mask"] = None
            if adv_labels is not None:
                adv_labels = adv_labels.to(self.device)
                forward_kwargs["adv_labels"] = adv_labels

            outputs = self.model(inputs, input_chans=self.input_chans, **forward_kwargs)
            
            # Normalize loss aggregation
            eval_loss += outputs["loss"].item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            if len(self.metrics) > 0:
                preds.extend(torch.argmax(outputs["logits"], dim=1).cpu().numpy().reshape(-1))
                gts.extend(labels.cpu().numpy().reshape(-1))

                probs = torch.softmax(outputs["logits"], dim=1)
                all_probs.extend(probs.cpu().detach().numpy())

        avg_eval_loss = eval_loss / total_samples

        if split == "val":
            self.history["eval_loss"].append(avg_eval_loss)
            self.history["eval_iters"].append(self.n_iters)
            self.writer.add_scalar("Loss/Eval", avg_eval_loss, self.n_iters)

        current_metrics = {}
        for metric in self.metrics:
            # Check if METRICS is defined in context, otherwise handle safely
            if metric in METRICS:
                if metric == "auroc":
                    prob_array = np.array(all_probs)
                    gt_array = np.array(gts)

                    if prob_array.shape[1] == 2:
                        val = METRICS[metric](gt_array, prob_array[:, 1])
                    else:
                        val = METRICS[metric](gt_array, prob_array)
                else:
                    val = METRICS[metric](np.array(gts), np.array(preds))
                
                if split == "val": 
                    self.eval_metrics[metric].append(val)
                else: 
                    self.test_metrics[metric] = val
                self.writer.add_scalar(f"Metrics/{split}/{metric}", val, self.n_iters)
                current_metrics[metric] = val

        if split == "val":
            print(f"[Eval @ Iter {self.n_iters}] Loss: {avg_eval_loss:.4f}, {current_metrics}")
            
            # Save standard checkpoint
            self._save_model(os.path.join(self.logdir, f"checkpoint_{self.n_iters}.pt"))

            # Logic to update "best_model.pt"
            target_val = current_metrics[self.patience_metric] if self.patience_metric != "loss" else avg_eval_loss
            # Flip logic for loss (lower is better) vs metrics (higher is better)
            if self.patience_metric == "loss":
                is_better = target_val < self.best_val
            else:
                is_better = target_val > self.best_val
                
            if is_better:
                self.best_val = target_val
                self.early_stop_counter = 0 # Reset patience
                self._save_model(self.best_checkpoint_path)
            else:
                self.early_stop_counter += 1
    
    def test_evaluate_classification(self, dataloader):
        self._load_model(self.best_checkpoint_path)
        self.model.eval()

        eval_loss, total_samples = 0.0, 0
        preds, gts = [], []
        all_probs = []

        for batch in dataloader:
            adv_labels = None
            channel_ids = None
            if type(batch) == dict:
                inputs = batch.pop("inputs")
                labels = batch.pop("labels")
                channel_ids = batch.pop("channel_ids", None)
                adv_labels = batch.pop("adv_labels", None)

            else:
                inputs, labels = batch

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            forward_kwargs = {"labels": labels}
            if channel_ids is not None:
                # Only add these if the dataset provided them (your new model)
                channel_ids = channel_ids.to(self.device)
                forward_kwargs["channel_ids"] = channel_ids
                forward_kwargs["padding_mask"] = None
            if adv_labels is not None:
                adv_labels = adv_labels.to(self.device)
                forward_kwargs["adv_labels"] = adv_labels

            outputs = self.model(inputs, input_chans=self.input_chans, **forward_kwargs)
            
            # Normalize loss aggregation
            eval_loss += outputs["loss"].item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            if len(self.metrics) > 0:
                preds.extend(torch.argmax(outputs["logits"], dim=1).cpu().numpy().reshape(-1))
                gts.extend(labels.cpu().numpy().reshape(-1))

                probs = torch.softmax(outputs["logits"], dim=1)
                all_probs.extend(probs.cpu().detach().numpy())

        avg_eval_loss = eval_loss / total_samples

        current_metrics = {}
        for metric in self.metrics:
            # Check if METRICS is defined in context, otherwise handle safely
            if metric in METRICS:
                if metric == "auroc":
                    prob_array = np.array(all_probs)
                    gt_array = np.array(gts)

                    if prob_array.shape[1] == 2:
                        val = METRICS[metric](gt_array, prob_array[:, 1])
                    else:
                        val = METRICS[metric](gt_array, prob_array)
                else:
                    val = METRICS[metric](np.array(gts), np.array(preds))
                current_metrics[metric] = val
        
        from collections import Counter
        print(f"Test Dataset label split: {Counter(gts)}")
        print(f"[Test @ Iter {self.n_iters}] Loss: {avg_eval_loss:.4f}, {current_metrics}")
        return current_metrics
