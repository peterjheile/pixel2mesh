import torch
import torch.nn.functional as F

def normal_loss(pred_normals, gt_normals, dist):
    """
    Args:
        pred_normals: [B, N, 3]
        gt_normals:   [B, M, 3]
        dist:         [B, N, M] – pairwise distances from chamfer
    """
    idx = dist.argmin(dim=2)  # [B, N]
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, 3)
    matched_gt_normals = torch.gather(gt_normals, 1, idx_exp)
    cos_sim = F.cosine_similarity(pred_normals, matched_gt_normals, dim=2)
    return (1 - cos_sim).mean()