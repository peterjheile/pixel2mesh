import os
import torch
import trimesh
import numpy as np
import pickle


def sample_gt_points(gt_path, num_points=1000, device="cpu"):

    ext = os.path.splitext(gt_path)[1].lower()

    if ext in [".obj", ".off", ".ply"]:
        mesh = trimesh.load(gt_path, process=False)
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return torch.tensor(points, dtype=torch.float32).to(device)

    elif ext == ".dat":
        with open(gt_path, "rb") as f:
            data = pickle.load(f, encoding='latin1')

        if isinstance(data, dict) and "points" in data:
            points = data["points"]
        elif isinstance(data, np.ndarray) and data.shape[1] >= 3:
            points = data[:, :3]
        else:
            raise ValueError("Unsupported .dat structure: missing 'points' or valid array")

        if points.shape[0] > num_points:
            idx = np.random.choice(points.shape[0], num_points, replace=False)
            points = points[idx]

        return torch.tensor(points, dtype=torch.float32).to(device)

    else:
        raise ValueError(f"Unsupported GT file type: {ext}")