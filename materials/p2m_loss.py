#Use to combine all of the loss functions together for a single block (remember there are three blocks)


import torch
import trimesh

from losses.chamfer_loss import chamfer_loss
from losses.laplacian_loss import laplacian_loss
from losses.edge_loss import edge_length_loss
from mesh.utils.sample_mesh import sample_mesh
from losses.normal_loss import normal_loss
from mesh.utils.graph_utlis import create_neighborhood_graph


def compute_p2m_loss(pred_verts, pred_normals, faces, 
                    gt_verts, gt_normals, device, 
                    num_mesh_samples = 1024,
                    weight_chamfer=1.0,
                    weight_normal=0.05,
                    weight_laplacian=0.1,
                    weight_edge=0.1
                    ):

    #now sure how large each mesh is so be able to set teh sample size. Of course the higher the 
    #more accurate the mode (but takes longer)
    gt_verts_sampled, gt_normals_sampled = sample_mesh(gt_verts, gt_normals, num_mesh_samples)
    edge_index = create_neighborhood_graph(faces)


    loss_chamfer, distances = chamfer_loss(pred_verts, gt_verts_sampled)
    loss_normal = normal_loss(pred_normals, gt_normals_sampled, distances)
    loss_edges = edge_length_loss(pred_verts, edge_index)
    loss_lap = laplacian_loss(pred_verts, edge_index)

    # Total loss (combine)
    total_loss = (
        weight_chamfer * loss_chamfer +
        weight_normal * loss_normal +
        weight_edge * loss_edges +
        weight_laplacian * loss_lap
    )

    loss_dict = {
        "chamfer": loss_chamfer.item(),
        "normal": loss_normal.item(),
        "edge": loss_edges.item(),
        "laplacian": loss_lap.item(),
        "total": total_loss.item()
    }

    return total_loss, loss_dict






