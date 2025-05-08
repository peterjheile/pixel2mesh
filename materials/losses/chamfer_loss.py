import torch

def chamfer_loss(pred_points, gt_points):
    dist = torch.cdist(pred_points, gt_points, p=2) ** 2  # [B, N, M]
    loss = dist.min(2)[0].mean(dim=1) + dist.min(1)[0].mean(dim=1)
    return loss.mean(), dist

    