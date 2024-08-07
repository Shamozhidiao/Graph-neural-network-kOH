import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_function(loss_type: str):
    loss_type = loss_type.strip().lower()
    if loss_type == 'l1':
        loss_fn = nn.L1Loss()
    elif loss_type == 'charbonnier':
        loss_fn = L1CharbonnierLoss()
    elif loss_type == 'l2':
        loss_fn = nn.MSELoss()
    else:
        raise NotImplementedError('please provide right loss type!')

    return loss_fn


class L1CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(torch.square(diff) + self.eps))
        return loss
