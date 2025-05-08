import torch

#samples the mesh
def sample_mesh(points, normals, num_samples):
        B, N, _ = points.shape
        idx = torch.randint(0, N, (B, num_samples), device=points.device)
        idx = idx.unsqueeze(-1).expand(-1, -1, 3)
        return torch.gather(points, 1, idx), torch.gather(normals, 1, idx)