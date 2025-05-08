## This page I used for testing
# I would load in a trained model, select me roughly 1000 testing images
#and either use them to find a accuracy, or just use on to visualize the results

import torch
from materials.p2m_model import Pixel2MeshModel
import trimesh
from torchvision import transforms
from PIL import Image
from datasetsP2M.dataset_shapenet import ShapeNetDataset 
from losses.chamfer_loss import chamfer_loss
from torch.utils.data import DataLoader
import numpy as np
from datasetsP2M.image_loader import load_image_tensor


if __name__ == "__main__":


    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Pixel2MeshModel().to(device)
    # model.eval()


    # # class_ids = ['02691156']  # airplane only
    # # train_dataset = ShapeNetDataset(root_dir=r"datasetsP2M\data\shapenet\shapenet_cleaned", class_ids=class_ids)
    # # test_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

    # # total_chamfer = 0.0

    # img_tensor = torch.randn(1, 3, 224, 224).to(device)
    # K = torch.tensor([[
    #     [250.0,   0.0, 112.0],  # fx,  0, cx
    #     [  0.0, 250.0, 112.0],  # 0,  fy, cy
    #     [  0.0,   0.0,   1.0]
    # ]], dtype=torch.float32) .to(device)
    # with torch.no_grad():
    #     outputs = model(img_tensor, K, device)

    # i = 0
    # chamfers = []
    # model.load_state_dict(torch.load(r"checkpoints\airplane_GCN_Features_Flipped\model_step_160000.pt"))
    # with torch.no_grad():
    #     for batch in test_loader:
    #         image = batch['image'].to(device)
    #         gt_points = batch['gt_points'].to(device)
    #         K = torch.eye(3, device=device).unsqueeze(0).repeat(image.shape[0], 1, 1)


    #         outputs = model(image, K, device)
    #         pred_verts = outputs["verts_final"]

    #         chamfer, _ = chamfer_loss(pred_verts, gt_points)

    #         chamfers.append(chamfer)
    #         i+=1
    #         if i == 500:
    #             break

    # chamfers = [c.item() if torch.is_tensor(c) else c for c in chamfers]
    # chamfers = np.array(chamfers)
    # min_c, max_c = chamfers.min(), chamfers.max()
    # normalized_chamfers = (chamfers - min_c) / (max_c - min_c)
    # avg_normalized_chamfer = normalized_chamfers.mean()


    # print(f"Normalized Mean Chamfer [0–1]: {avg_normalized_chamfer:.4f}")



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Pixel2MeshModel().to(device)
    model.eval()

    img_tensor = torch.randn(1, 3, 224, 224).to(device)
    K = torch.tensor([[
        [250.0,   0.0, 112.0],  # fx,  0, cx
        [  0.0, 250.0, 112.0],  # 0,  fy, cy
        [  0.0,   0.0,   1.0]
    ]], dtype=torch.float32) .to(device)


    with torch.no_grad():
        outputs = model(img_tensor, K, device)

    model.load_state_dict(torch.load(r"checkpoints\airplane_stage_1_big\model_step_5000.pt"))


    img_tensor = load_image_tensor(r"datasetsP2M\data\shapenet\shapenet_cleaned\02691156\1b7ac690067010e26b7bd17e458d0dcb\00.png").to(device)

    with torch.no_grad():
        outputs = model(img_tensor, K, device)
        
    verts1 = outputs["verts_stage2"]
    faces1 = outputs["faces_stage2"]
    normals1 = outputs["normals_stage3"]

    verts2 = outputs["verts_stage3"]
    faces2 = outputs["faces_stage3"]
    normals2 = outputs["normals_stage3"]


    verts3 = outputs["verts_final"]
    faces3 = outputs["faces_final"]
    normals3 = outputs["normals_final"]

    #stage opne of the mesh
    mesh1 = trimesh.Trimesh(vertices=verts1.cpu().numpy(), 
                        faces=faces1.cpu().numpy(), 
                        vertex_normals=normals1.cpu().numpy())

    mesh1.show()


    #stage 2 of the mesh
    mesh2 = trimesh.Trimesh(vertices=verts2.cpu().numpy(), 
                        faces=faces2.cpu().numpy(), 
                        vertex_normals=normals2.cpu().numpy())

    mesh2.show()


    #stage 3 of the mesh
    mesh3 = trimesh.Trimesh(vertices=verts3.cpu().numpy(), 
                        faces=faces3.cpu().numpy(), 
                        vertex_normals=normals3.cpu().numpy())

    mesh3.show()








