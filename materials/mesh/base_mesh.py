'''
I just use this wrapper function to create the intial base mesh for the Mesh Deformation
Blocks. Specifically the first mesh deformation block. 

This generates a round ellipsoid mesh. Changing the level will decided how many vertices/faces are
generated. I opt for subdivisions=2 as it creates 162 vertices and 320 faces (closest to the p2m paper implemt)

'''

import trimesh
import torch

def get_base_mesh(subdivisions=2, scale = (1.0, 1.0, 0.8), device="cpu"):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    mesh.vertices *= scale

    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)


    return verts, faces