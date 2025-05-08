import torch

#no batching
def laplacian_loss(verts, edge_index):
    V = verts.shape[0]
    device = verts.device
    src, dst = edge_index

    lap = torch.zeros_like(verts)
    deg = torch.zeros(V, 1, device=device)

    lap.index_add_(0, src, verts[dst])
    deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float32).unsqueeze(1)) 

    lap = lap / deg.clamp(min=1)
    diff = verts - lap
    loss = (diff ** 2).sum(dim=1).mean()

    return loss