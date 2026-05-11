import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):

    def __init__(
            self,
            tasktype: str="classification",
            *args,
            **kwargs
            ):

        super(BaseModel, self).__init__()
        self.tasktype = tasktype

        if tasktype == "classification":
            self.loss_function = self.classification_loss
        else:
            raise ValueError(f"Unknown tasktype: {tasktype} passed. Select among ['classification']")


    def forward(self, x):
        raise NotImplementedError
    
    def classification_loss(self, logits=None, labels=None, **kwargs):
        """
            logits: (B, n_classes) - model predictions without the application of softmax
            labels: (B) - ground truth labels
        """
        
        B = logits.shape[0]
        n_classes = logits.shape[-1]
        if logits.ndim > 2:
            logits = logits.view(-1, n_classes)
            labels = labels.view(-1)

        return F.cross_entropy(logits, labels, **kwargs)

    def get_optimizer_params(self, weight_decay: float):
        """
        Groups parameters into decay and no-decay categories.
        Excludes classifier, biases, and normalization layers from weight decay.
        """
        decay = []
        no_decay = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            # Exclude classifier AND small parameters like biases/norms
            if "classifier" in name or "bias" in name or "norm" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}
        ]
    








