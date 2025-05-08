'''

Any utility function required for constructing the neighborhood graph I will put in here.
A neighborhood graph is created so that adjacent vertices can be linked to form the Graph Network.

'''

import torch

#NOTE: this is faster, but I don't account for duplicate edges. It should
#be okay as GCNs hand duplicates pretty alright; hwoever it should be changed letter optimally.
def create_neighborhood_graph(faces):
    if not isinstance(faces, torch.Tensor):
        faces = torch.tensor(faces, dtype=torch.long)

    i0 = faces[:, [0, 1]]
    i1 = faces[:, [1, 2]]
    i2 = faces[:, [2, 0]]

    edges = torch.cat([i0, i1, i2], dim=0)
    edges_rev = edges[:, [1, 0]]

    all_edges = torch.cat([edges, edges_rev], dim=0)
    return all_edges.t().contiguous()