import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import trimesh
import pickle
from collections import defaultdict


class ShapeNetDataset(Dataset):
    def __init__(self, root_dir, class_ids, transform=None):

        self.samples = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #Resenet's ImageNet mean and std to help the Resent Model
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        for class_id in class_ids:
            class_path = os.path.join(root_dir, class_id)
            if not os.path.isdir(class_path):
                continue
            for instance_id in os.listdir(class_path):
                instance_path = os.path.join(class_path, instance_id)
                obj_path = os.path.join(instance_path, "model.obj")
                for i in range(5):  # 5 views
                    img_path = os.path.join(instance_path, f"{i:02d}.png")
                    dat_path = os.path.join(instance_path, f"{i:02d}.dat")
                    if os.path.exists(img_path) and os.path.exists(dat_path):
                        self.samples.append({
                            "image": img_path,
                            "dat": dat_path,
                            "obj": obj_path,
                            "class_id": class_id,
                            "instance_id": instance_id
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["image"]).convert("RGB")
        image = self.transform(image)

        # Load .dat
        with open(sample["dat"], "rb") as f:
            data = pickle.load(f, encoding='latin1')
            
        gt_points = torch.tensor(data[:, :3], dtype=torch.float32)
        gt_normals = torch.tensor(data[:, 3:6], dtype=torch.float32)

        # Load .obj
        mesh = trimesh.load_mesh(sample["obj"], process=False)
        verts = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.long)

        return {
            "image": image,
            "gt_points": gt_points,
            "gt_normals": gt_normals,
            "K": torch.eye(3),
            "verts": verts,
            "faces": faces,
            "meta": {
                "class_id": sample["class_id"],
                "instance_id": sample["instance_id"],
                "obj_path": sample["obj"]
            }
        }
    
    def __str__(self):
        class_counts = defaultdict(set)
        for s in self.samples:
            class_counts[s["class_id"]].add(s["instance_id"])

        summary = [f"ShapeNetDataset with {len(self.samples)} samples"]
        for cid, instances in class_counts.items():
            summary.append(f"  {cid}: {len(instances)} instances, {len(instances)*5} views")
        return "\n".join(summary)