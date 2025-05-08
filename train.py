import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from p2m_loss import compute_p2m_loss
from torch.utils.data import DataLoader
from p2m_model import Pixel2MeshModel
from datasetsP2M.dataset_shapenet import ShapeNetDataset


def train(model, dataloader, optimizer, num_epochs, device, save_every_batches=1000, log_dir="runs/", save_dir="checkpoints/"):
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    model.train()

    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        start_time = time.time()

        #Note, the batch is shuffled in each iteration
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            images = batch['image'].to(device)
            gt_points = batch['gt_points'].to(device)
            gt_normals = batch['gt_normals'].to(device)

            K = torch.eye(3, device=device).unsqueeze(0).repeat(images.shape[0], 1, 1)
            outputs = model(images, K, device)

            #definet the loss weights for each stage.
            total_loss = 0
            stage_weights = [1, 0, 0]
            loss_weights = [[1, 0.0, 100, 100], [1, 5, 100, 30], [3, 10, 100, 10]]
            loss_names = ['stage2', 'stage3', 'final']
            loss_dict_total = {}

            for i, stage in enumerate(loss_names):
                verts = outputs[f'verts_{stage}']
                faces = outputs[f'faces_{stage}']
                normals = outputs[f'normals_{stage}']
                chamfer = loss_weights[i][0]
                normal = loss_weights[i][1]
                laplacian = loss_weights[i][2]
                edge = loss_weights[i][3]
                num_mesh_samples = 4096


                loss, loss_dict = compute_p2m_loss(
                    pred_verts=verts,
                    pred_normals=normals,
                    faces=faces,
                    gt_verts=gt_points,
                    gt_normals=gt_normals,
                    device=device,
                    weight_chamfer=chamfer,
                    weight_edge=edge,
                    weight_laplacian=laplacian,
                    weight_normal=normal,
                    num_mesh_samples=num_mesh_samples
                )

                weighted_loss = stage_weights[i] * loss
                total_loss += weighted_loss

                for k, v in loss_dict.items():
                    writer.add_scalar(f"Loss/{stage}_{k}", v, global_step)

                
            total_loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/Total", total_loss.item(), global_step)
            epoch_loss += total_loss.item()
            global_step += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {total_loss.item():.4f}")

            if global_step % save_every_batches == 0:
                save_path = os.path.join(save_dir, f"model_step_{global_step}.pt")
                torch.save(model.state_dict(), save_path)
                print(f"💾 Saved model at {save_path} (step {global_step})")

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Time: {time.time() - start_time:.2f}s")



    writer.close()


if __name__ == '__main__':
    batch_size = 1
    num_epochs = 50
    lr = 1e-3
    log_dir="runs/airplane_stage_1_big"
    save_dir="checkpoints/airplane_stage_1_big"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    save_every_batches=5000

    #load the dataset from the dataset folder
    class_ids = ['02691156']  #airplane calss only
    train_dataset = ShapeNetDataset(root_dir=r"datasetsP2M\data\shapenet\shapenet_cleaned", class_ids=class_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    #NOTE: Here, depending on what I was testing, I would sometimes split into a train and test set
    #I removed this portion as I tried differnt methods later. However, the train_loader
    #can be easily subscripted to generate a train and a testing set.

    model = Pixel2MeshModel()


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    train(model, train_loader, optimizer, num_epochs, device, save_every_batches, log_dir, save_dir)

