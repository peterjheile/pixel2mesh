import os
import shutil


img_dat_dir = r"datasetsP2M\data\shapenet\data_tf"
obj_dir = r'datasetsP2M\data\shapenet\shapenet_extracted'
output_dir = r"datasetsP2M\data\shapenet\shapenet_cleaned"
os.makedirs(output_dir, exist_ok=True)

classes = [
    "02691156",
    # "02958343",
    "03001627",
    # "03797390",
    # "04379243",
]


# ['02691156', '03001627']

def find_object_file(base_dir):
    direct_path = os.path.join(base_dir, "model_normalized.obj")

    if os.path.isfile(direct_path):
        return direct_path

    # Check inside the models/ subdirectory
    nested_path = os.path.join(base_dir, "models", "model_normalized.obj")
    if os.path.isfile(nested_path):
        return nested_path

    # Not found
    return False




def clean_dataset(classes, obj_dir, img_dat_dir, output_dir, instance_count = 5000):
    for class_name in classes:
        instance_tracker = 0
        img_dat_instance_dir = os.path.join(img_dat_dir, class_name)

        if class_name in os.listdir(img_dat_instance_dir):
            print("Class Already Cleaned")
            continue

        os.makedirs(img_dat_instance_dir, exist_ok=True)


        for instance_id in os.listdir(img_dat_instance_dir):

            img_data_instance_solo = os.path.join(img_dat_instance_dir, instance_id, "rendering")
            img_destination_instance_solo = os.path.join(output_dir, class_name, instance_id)


            for file in os.listdir(img_data_instance_solo):

                #check if corresponding .obj
                obj_path = find_object_file(os.path.join(obj_dir, class_name, instance_id))

                if file.endswith('.png') or file.endswith('.dat') and obj_path:

                    os.makedirs(img_destination_instance_solo, exist_ok=True)
                    shutil.copy(
                        os.path.join(img_data_instance_solo, file),
                        os.path.join(img_destination_instance_solo, file)
                    )
                    shutil.copy(
                        obj_path,
                        os.path.join(img_destination_instance_solo, "model.obj")
                    )

            instance_tracker += 1
            if instance_tracker == 5000:
                break 

        print(f"{instance_tracker} instances of goal {instance_count} instances saved for class {class_name}")



clean_dataset(classes, obj_dir, img_dat_dir, output_dir)



