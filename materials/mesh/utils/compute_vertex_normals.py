import trimesh
import torch

#the vertex normals for each mesh deformation stage are computed here.
def compute_vertex_normals(verts, faces):
    mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(),
                           faces=faces.detach().cpu().numpy(),
                           process=False)
    normals = mesh.vertex_normals
    return torch.tensor(normals, dtype=torch.float32, device=verts.device)