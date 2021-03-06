import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionLoss(nn.Module):
    """
    Cross entropy loss between two permutations.
    """
    def __init__(self):
        super(RegressionLoss, self).__init__()

    def forward(self, pred_perm, gt_perm, n1_ns, n2_ns):
        batch_num = pred_perm.shape[0]

        pred_perm = pred_perm.to(dtype=torch.float32)

        assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        loss = torch.tensor(0.).to(pred_perm.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            loss += F.binary_cross_entropy(
                pred_perm[b, :n1_ns[b], :n2_ns[b]],
                gt_perm[b, :n1_ns[b], :n2_ns[b]],
                reduction='sum')
            n_sum += n1_ns[b].to(n_sum.dtype).to(pred_perm.device)

        return loss / n_sum
