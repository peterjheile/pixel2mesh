import trimesh
import torch


#used trimesh to subdivide the mesh. This is one of teh few batching bottlenecks, as 
#trimesh does not allow abtched mesh deformation
def subdivide_mesh(verts, faces):
    mesh = trimesh.Trimesh(
        vertices=verts.detach().cpu().numpy(),
        faces=faces.detach().cpu().numpy()
    )

    subdivided = mesh.subdivide()

    new_verts = torch.tensor(subdivided.vertices, dtype=torch.float32)
    new_faces = torch.tensor(subdivided.faces, dtype=torch.long)

    return new_verts, new_faces