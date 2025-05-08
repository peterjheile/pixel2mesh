import torch

#does not work for batching
def edge_length_loss(verts, edge_index):
    src, dst = edge_index
    v_src = verts[src] 
    v_dst = verts[dst]  

    edge_lengths = (v_src - v_dst).norm(dim=1)
    return edge_lengths.pow(2).mean()