'''
In this file I have all utility functions that map 3d vertices to the corresponding feature plane. 
This allows each vertice to be associated with features from the CNN.

'''

import torch


def project_vertices_to_image(verts, K, image_size, feature_map_size, device):
    V = verts.shape[0]

    #project to image plane using K
    ones = torch.ones((V, 1), device=device)
    verts_hom = torch.cat([verts, ones], dim=1)
    verts_3d = verts_hom[:, :3]


    proj = verts_3d @ K.T

    uv = proj[:, :2] / proj[:, 2:3]

    H_img, W_img = image_size
    uv_norm = uv.clone()
    uv_norm[:, 0] /= W_img
    uv_norm[:, 1] /= H_img

    #scale to the sized of the feature map
    H_feat, W_feat = feature_map_size
    uv_feat = uv_norm.clone()
    uv_feat[:, 0] *= W_feat
    uv_feat[:, 1] *= H_feat

    return uv_feat