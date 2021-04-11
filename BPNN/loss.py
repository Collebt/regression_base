import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionLoss(nn.Module):
    """
    Cross entropy loss between two permutations.
    """
    def __init__(self):
        super(RegressionLoss, self).__init__()

    def forward(self, x_pred, x_gt, Q_pred, Q_gt):

        loss = (x_pred - x_gt) ** 2 + (Q_pred - Q_gt)
        return loss 
