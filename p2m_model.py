import torch
import torch.nn as nn
from models.gcn_deformer import GCNDeformer
from mesh.utils.graph_utlis import create_neighborhood_graph
from mesh.base_mesh import get_base_mesh
from mesh.utils.subdivide_mesh_utils import subdivide_mesh
from mesh.utils.feature_sampling_utils import sample_vertex_features_from_map
from mesh.utils.projection_utils import project_vertices_to_image
from mesh.utils.graph_utlis import create_neighborhood_graph
from mesh.utils.projection_utils import project_vertices_to_image
from models.feature_extractor import FeatureExtractor
from mesh.utils.compute_vertex_normals import compute_vertex_normals



class Pixel2MeshModel(nn.Module):
    def __init__(self, image_size=(224, 224), hidden_dim=256, num_blocks=3):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_size = image_size
        self.feature_extractor = FeatureExtractor(backbone='resnet18').to(self.device)

        self.stage1 = None
        self.stage2 = None
        self.stage3 = None

        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

    def init_stage(self, in_dim, edge_index):
        return GCNDeformer(
            in_dim=in_dim,
            hidden_dim=self.hidden_dim,
            out_dim=3,
        ).to(self.device)
    
    def forward(self, image, K, device):

        if K.dim() == 3:
            K = K.squeeze(0)
        K = K.to(device)
        

        features = self.feature_extractor(image)


        verts1, faces1 = get_base_mesh(subdivisions=2, device=device)
        verts1[:, 2] += 2.0 
        edge_index1 = create_neighborhood_graph(faces1).to(device)

        uv1 = project_vertices_to_image(verts1, K, self.image_size, features['layer3'].shape[-2:], device)
        feat1 = sample_vertex_features_from_map(features['layer3'], uv1).to(device) 


        if not hasattr(self, 'feat_proj1'):
            self.feat_proj1 = nn.Linear(256, 128).to(device)
        feat1 = self.feat_proj1(feat1)

        if self.stage1 is None:
            self.stage1 = self.init_stage(in_dim=128, edge_index=edge_index1)

        offsets1 = self.stage1(feat1, edge_index1)
        verts2 = verts1 + offsets1
        normals1 = compute_vertex_normals(verts1, faces1)


        verts2, faces2 = subdivide_mesh(verts2, faces1)
        verts2 = verts2.to(self.device)
        faces2 = faces2.to(device)
        edge_index2 = create_neighborhood_graph(faces2).to(device)

        uv2 = project_vertices_to_image(verts2, K, self.image_size, features['layer4'].shape[-2:], device)
        feat2 = sample_vertex_features_from_map(features['layer4'], uv2).to(device)

        if not hasattr(self, 'feat_proj2'):
            self.feat_proj2 = nn.Linear(512, 256).to(device)
        feat2 = self.feat_proj2(feat2)

        if self.stage2 is None:
            self.stage2 = self.init_stage(in_dim=256, edge_index=edge_index2)

        offsets2 = self.stage2(feat2, edge_index2)
        verts3 = verts2 + offsets2
        normals2 = compute_vertex_normals(verts2, faces2)

        verts3, faces3 = subdivide_mesh(verts3, faces2)
        verts3 = verts3.to(device)
        faces3 = faces3.to(device)
        edge_index3 = create_neighborhood_graph(faces3).to(device)

        uv3 = project_vertices_to_image(verts3, K, self.image_size, features['layer4'].shape[-2:], device)
        feat3 = sample_vertex_features_from_map(features['layer4'], uv3).to(device)  # [V, 128]

        if not hasattr(self, 'feat_proj3'):
            self.feat_proj3 = nn.Linear(512, 256).to(device)
        feat3 = self.feat_proj3(feat3)

        if self.stage3 is None:
            self.stage3 = self.init_stage(in_dim=256, edge_index=edge_index3)

        offsets3 = self.stage3(feat3, edge_index3)
        normals3 = compute_vertex_normals(verts3, faces3)
        verts_final = verts3 + offsets3

        normals_final = compute_vertex_normals(verts_final, faces3)

        #NOTE: Stage 1 is literally the initial ellipsoid mesh that we have created, unaltered.
        return {
            "verts_stage1": verts1,
            "faces_stage1": faces1,
            "normals_stage1": normals1,

            "verts_stage2": verts2,
            "faces_stage2": faces2,
            "normals_stage2": normals2,

            "verts_stage3": verts3,
            "faces_stage3": faces3,
            "normals_stage3": normals3,

            "verts_final": verts_final,
            "faces_final": faces3,
            "normals_final": normals_final,
        }