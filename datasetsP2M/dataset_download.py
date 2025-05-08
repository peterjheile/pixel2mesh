import os
import shutil
import zipfile
import tempfile
from huggingface_hub import hf_hub_download

class_ids = [
    "02691156",  # Airplane
    "02958343",  # Car
    "03001627",  # Chair
    "03797390",  # Mug
    "04379243",  # Table
]

zip_dir = "datasetsP2M/data/shapenet_zips"
os.makedirs(zip_dir, exist_ok=True)


extract_dir = "datasetsP2M/data/shapenet_extracted"
os.makedirs(extract_dir, exist_ok=True)

delete_zip_after_extract = False

for class_id in class_ids:
    print(f"\nDownloading {class_id}.zip from Hugging Face...")


    zip_path = hf_hub_download(
        repo_id="ShapeNet/ShapeNetCore",
        filename=f"{class_id}.zip",
        repo_type="dataset"
    )


    local_zip_path = os.path.join(zip_dir, f"{class_id}.zip")
    shutil.copy(zip_path, local_zip_path)
    print(f"Saved ZIP to: {local_zip_path}")


    class_extract_path = os.path.join(extract_dir, class_id)
    os.makedirs(class_extract_path, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_extract_dir:
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_extract_dir)

        inner_dir = os.path.join(tmp_extract_dir, class_id)
        if not os.path.exists(inner_dir):
            raise RuntimeError(f"Expected folder {inner_dir} inside ZIP")


        for instance_id in os.listdir(inner_dir):
            src = os.path.join(inner_dir, instance_id)
            dst = os.path.join(class_extract_path, instance_id)
            shutil.move(src, dst)

        print(f"Extracted to: {class_extract_path}")


    if delete_zip_after_extract:
        os.remove(local_zip_path)
        print(f"Deleted ZIP: {local_zip_path}")

print("\nAll classes downloaded and extracted cleanly.")