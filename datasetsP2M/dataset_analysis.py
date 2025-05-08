import os

extract_dir = r"datasetsP2M\data\shapenet\shapenet_extracted"

class_names = {
    "02691156": "Airplane",
    "02958343": "Car",
    "03001627": "Chair",
    "03797390": "Mug",
    "04379243": "Table",
}


class_counts = {}

for class_id in sorted(os.listdir(extract_dir)):
    class_path = os.path.join(extract_dir, class_id)
    if not os.path.isdir(class_path):
        continue

    instance_folders = [
        f for f in os.listdir(class_path)
        if os.path.isdir(os.path.join(class_path, f))
    ]
    
    class_counts[class_id] = len(instance_folders)

#Print summary
print("📊 Instance counts per class:")
for class_id, count in class_counts.items():
    print(f"  {class_names[class_id]}: {count} instances")

total = sum(class_counts.values())
print(f"\nTotal instances across all classes: {total}")
