import os
import pandas as pd

# Define dataset paths and subfolder lists
dataset_paths = {
    "fan": ["1024_256", "1024_512", "2048_256", "2048_512", "4096_256", "4096_512"],
    "pump": ["1024_256", "1024_512", "2048_256", "2048_512", "4096_256", "4096_512"],
    "slider": ["1024_256", "1024_512", "2048_256", "2048_512", "4096_256", "4096_512"],
    "valve": ["1024_256", "1024_512", "2048_256", "2048_512", "4096_256", "4096_512"]
}

base_dataset_path = "D:\\MIMII\\logmel_"
output_folder = "D:\\MIMII\\folder_csv"  # Output folder for CSV files
os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists

subfolder_lists = {
    "fan": ['0_db_fan', '6_db_fan', 'm6_db_fan'],
    "pump": ['0_db_pump', '6_db_pump', 'm6_db_pump'],
    "slider": ['0_db_slider', '6_db_slider', 'm6_db_slider'],
    "valve": ['0_db_valve', '6_db_valve', 'm6_db_valve']
}

# Function to create path and label lists
def create_path_and_label(dataset_path, subfolder_list):
    path_list = []
    label_list = []
    for subfolder in subfolder_list:
        folder_path = os.path.join(dataset_path, subfolder)
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.png'):
                    path_list.append(os.path.join(root, file))
                    label = os.path.basename(root).split('/')[-1]
                    label_list.append(0 if label == 'normal' else 1)
    return path_list, label_list

# Function to save paths and labels to a CSV
def save_to_csv(dataset_path, output_file, subfolder_list):
    path_list, label_list = create_path_and_label(dataset_path, subfolder_list)
    df = pd.DataFrame(list(zip(path_list, label_list)), columns=['path', 'label'])
    df.to_csv(output_file, index=False)

# Iterate over all combinations of datasets and subfolder lists
for category, config_list in dataset_paths.items():
    for config in config_list:
        dataset_path = f"{base_dataset_path}{config}"
        output_file = os.path.join(output_folder, f"{category}_{config}.csv")
        save_to_csv(dataset_path, output_file, subfolder_lists[category])
