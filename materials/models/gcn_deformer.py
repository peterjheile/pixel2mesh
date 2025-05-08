#############
#
# This is the core Graph convolution that I uyse for each Mesh deformation block. 
#
#############


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = GCNConv(dim, dim)
        self.conv2 = GCNConv(dim, dim)

    def forward(self, x, edge_index):
        identity = x
        out = F.relu(self.conv1(x, edge_index))
        out = self.conv2(out, edge_index)
        return F.relu(out + identity)
    


class GCNDeformer(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, out_dim=3, num_blocks=3):
        super().__init__()
        self.input_proj = GCNConv(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([GResBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_proj = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.input_proj(x, edge_index))
        for block in self.blocks:
            x = block(x, edge_index)
        return self.output_proj(x, edge_index)