#######################################################################################
#
#   For the most part, I just use this main function as a testing page for any utils, scripts
#   and functions write.
#
#
#######################################################################################


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import torch
from torch.utils.data import DataLoader
from p2m_model import Pixel2MeshModel
from datasetsP2M.dataset_shapenet import ShapeNetDataset
from p2m_loss import compute_p2m_loss
from train import train
from PIL import Image
from torchvision import transforms
import open3d as o3d
import trimesh

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Pixel2MeshModel()
    # model.load_state_dict(torch.load(r"checkpoints\airplane_only\model_step_8000.pt", map_location=device))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    #load and process image
    img_path = r"datasetsP2M\data\shapenet\shapenet_cleaned\02691156\22829f20e331d563dd455eb19d4d269e\00.png"
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    K = torch.eye(3, device=device).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image, K, device)

    model.load_state_dict(torch.load(r"checkpoints\airplane_only\model_step_1000.pt", map_location=device))

    #You can access outputs like:
    verts = outputs["verts_final"]
    faces = outputs["faces_final"]
    normals = outputs["normals_final"]


    mesh = trimesh.Trimesh(vertices=verts.cpu().numpy(), 
                        faces=faces.cpu().numpy(), 
                        vertex_normals=normals.cpu().numpy())

    mesh.show()

    # # === Config ===
    # batch_size = 1
    # num_epochs = 50
    # lr = 1e-4
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # === Dataset & Dataloader ===
    # class_ids = ['02691156', '03001627']  # airplane, car for example
    # train_dataset = ShapeNetDataset(root_dir=r"datasetsP2M\data\shapenet\shapenet_cleaned", class_ids=class_ids)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # # === Model ===
    # model = Pixel2MeshModel()  # Already returns verts, normals, and has edge_index

    # # === Optimizer ===
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # # # === Train ===
    # # train(model, train_loader, optimizer, num_epochs, device)

    # batch = next(iter(train_loader))
    # image = batch["image"].to(device)      # shape: [1, 3, 224, 224]
    # K = batch["K"].to(device)              # shape: [1, 3, 3]
    # model.eval()
    # with torch.no_grad():
    #     output = model(image, K, device)

    # print(output.keys())



    # image_path = r"datasetsP2M\data\shapenet\shapenet_cleaned\02691156\1a9b552befd6306cc8f2d5fe7449af61\00.png"
    # transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),
    # ])
    
    # img = Image.open(image_path).convert("RGB")
    # img_tensor = transform(img).unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]
    # K = torch.eye(3)

    # model.eval()
    # with torch.no_grad():
    #     results = model(img_tensor, K, device)

    # print(results.keys())
    
    # verts, faces, normals = output['verts_stage2'], output['faces_stage2'], output['normals']

    # # print("Verts:", verts.shape)     # [1, V, 3]
    # # print("Normals:", normals.shape) # [1, V, 3]
    # # print("Faces:", faces.shape)     # [F, 3]

    # verts_np = verts.cpu().numpy()     # [2562, 3]
    # faces_np = faces.cpu().numpy()    # [2562, 3]

    # # mesh_o3d = o3d.geometry.TriangleMesh()
    # # mesh_o3d.vertices = o3d.utility.Vector3dVector(verts_np)
    # # mesh_o3d.triangles = o3d.utility.Vector3iVector(faces_np)

    # # # Optional: set precomputed normals
    # # if normals is not None:
    # #     mesh_o3d.vertex_normals = o3d.utility.Vector3dVector(normals.cpu().numpy())
    # # else:
    # #     mesh_o3d.compute_vertex_normals()

    # # # Visualize
    # # o3d.visualization.draw_geometries([mesh_o3d])

    # # print(faces_np.shape)

    # # Create the mesh
    # mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)

    # # Optional: assign vertex normals if you have them
    # # if normals is not None:
    # #     mesh.vertex_normals = normals.cpu().numpy()

    # # Visualize
    # mesh.show()

    # # Create point cloud
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(verts_np)
    # pcd.normals = o3d.utility.Vector3dVector(normals_np)

    # # Optional: normalize for consistent arrow size
    # pcd.normalize_normals()

    # # Visualize with normals shown as arrows
    # o3d.visualization.draw_geometries(
    #     [pcd],
    #     point_show_normal=True,
    #     window_name="Vertex Point Cloud + Normals"
    # )



    # # Print output shapes
    # print("Predicted vertices:", verts.shape)   # Expect [1, V, 3]
    # print("Predicted normals:", normals.shape)  # Expect [1, V, 3]
    # print("Mesh faces:", faces.shape)           # Expect [F, 3]