'''

In this file contains the util functions for smapling the vertex's corresponding features
from each feature map of the Feature Extractor.

'''

import torch
import torch.nn.functional as F

def normalize_grid_coords(uv, feat_map_shape):
    H, W = feat_map_shape
    uv_norm = uv.clone()
    uv_norm[:, 0] = (uv[:, 0] / (W - 1)) * 2 - 1
    uv_norm[:, 1] = (uv[:, 1] / (H - 1)) * 2 - 1
    return uv_norm


def sample_vertex_features_from_map(feat_map, uv_feat):
    device = feat_map.device
    B, C, H, W = feat_map.shape
    V = uv_feat.shape[0]

    uv_feat = uv_feat.to(device)
    grid_norm = normalize_grid_coords(uv_feat, (H, W)).to(device) 
    grid = grid_norm.view(1, V, 1, 2)

    sampled = F.grid_sample(
        feat_map, grid,
        mode='bilinear',
        align_corners=True
    )

    return sampled.squeeze(-1).squeeze(0).transpose(0, 1)