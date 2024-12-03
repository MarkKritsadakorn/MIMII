import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import os
import time
import json

# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        image = Image.open(img_path).convert("RGB")  # Ensure 3 channels
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# Data Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Paths to CSV files for each type
file_paths1 = [
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/fan_1024_256.csv',
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/fan_1024_512.csv',
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/fan_2048_256.csv',
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/fan_2048_512.csv',
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/fan_4096_256.csv',
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/fan_4096_512.csv'
]

file_paths2 = [
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/pump_1024_256.csv',
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/pump_1024_512.csv',
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/pump_2048_256.csv',
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/pump_2048_512.csv',
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/pump_4096_256.csv',
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/pump_4096_512.csv'
]

file_paths3 = [
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/slider_1024_256.csv',
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/slider_1024_512.csv',
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/slider_2048_256.csv',
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/slider_2048_512.csv',
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/slider_4096_256.csv',
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/slider_4096_512.csv'
]

file_paths4 = [
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/valve_1024_256.csv',
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/valve_1024_512.csv',
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/valve_2048_256.csv',
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/valve_2048_512.csv',
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/valve_4096_256.csv',
    r'/home/eecommu06/Desktop/New Folder/logmel_png/folder_csv/valve_4096_512.csv'
]

# Output directories for each type
output_dir1 = "fan_results"
output_dir2 = "pump_results"
output_dir3 = "slider_results"
output_dir4 = "valve_results"

# Hyperparameters
batch_size = 32
num_epochs = 30
learning_rate = 0.001

# Function to handle the training and evaluation
def train_and_evaluate(file_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save hyperparameters in a readable format
    hyperparams = f"""
    batch_size = {batch_size}
    num_epochs = {num_epochs}
    learning_rate = {learning_rate}
    """
    with open(os.path.join(output_dir, "hyperparameters.txt"), "w") as f:
        f.write(hyperparams)
    print(f"Hyperparameters saved at {os.path.join(output_dir, 'hyperparameters.txt')}")

    # Variables to save fold results and overall metrics
    fold_results = []
    all_val_auc = []

    # Loop through each file
    for fold, file_path in enumerate(file_paths):
        print(f"Processing file {fold + 1}/{len(file_paths)}: {file_path}")
        
        # DataLoader
        dataset = CustomImageDataset(csv_file=file_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Define CNN Model
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                self.fc1 = nn.Linear(32 * 32 * 32, 128)
                self.fc2 = nn.Linear(128, 1)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = self.relu(self.fc1(x))
                x = self.sigmoid(self.fc2(x))
                return x
        
        model = SimpleCNN()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Variables to store results
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_f1": [],
            "val_auc": []
        }
        all_targets = []
        all_preds = []
        start_time = time.time()

        # Training Loop
        for epoch in range(num_epochs):  # 10 epochs for simplicity
            model.train()
            train_loss = 0
            correct_train = 0
            total_train = 0
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)  # Move data to GPU
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                # Training accuracy
                preds = (outputs > 0.5).float()
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)
            
            train_loss /= len(dataloader)
            train_acc = correct_train / total_train
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_loss = 0
                all_labels = []
                all_preds_epoch = []
                for images, labels in dataloader:
                    images, labels = images.to(device), labels.to(device).unsqueeze(1)  # Move data to GPU
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    preds = (outputs > 0.5).float()
                    all_labels.extend(labels.cpu().numpy())
                    all_preds_epoch.extend(preds.cpu().numpy())
                
                val_loss /= len(dataloader)
                all_labels = torch.tensor(all_labels)
                all_preds_epoch = torch.tensor(all_preds_epoch)
                val_acc = (all_preds_epoch == all_labels).float().mean().item()
                val_f1 = f1_score(all_labels, all_preds_epoch, average="binary")
                val_auc = roc_auc_score(all_labels, all_preds_epoch)
                
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["train_acc"].append(train_acc)
                history["val_acc"].append(val_acc)
                history["val_f1"].append(val_f1)
                history["val_auc"].append(val_auc)
                
                all_targets.extend(all_labels.numpy())
                all_preds.extend(all_preds_epoch.numpy())

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")

        # Save the model after all epochs
        model_name = os.path.basename(file_path).replace('.csv', '.pth')
        model_save_path = os.path.join(output_dir, model_name)
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")

        # Save processing time to txt file
        end_time = time.time()
        processing_time = end_time - start_time
        with open(os.path.join(output_dir, "processing_time.txt"), "a+") as f:
            f.write(f"{os.path.basename(file_path).replace('.csv', '')}\t{processing_time}\n")
        print(f"Processing time saved for {os.path.basename(file_path).replace('.csv', '')}")

        # Save fold results
        history_path = os.path.join(output_dir, f"{os.path.basename(file_path).replace('.csv', '')}_history.csv")
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f"Saved training history for {os.path.basename(file_path)} at {history_path}")
        
        # Save confusion matrix data
        cm_results = {
            "true_labels": [int(label) for label in all_targets],  # Ensure labels are integers
            "predicted_labels": [int(pred) for pred in all_preds]  # Ensure predictions are integers
        }
        cm_results_path = os.path.join(output_dir, f"{os.path.basename(file_path).replace('.csv', '')}_confusion_matrix_data.csv")
        pd.DataFrame(cm_results).to_csv(cm_results_path, index=False)
        print(f"Saved confusion matrix data for {os.path.basename(file_path)} at {cm_results_path}")

        # Record fold results summary
        fold_results.append({"file": os.path.basename(file_path), "val_f1": max(history["val_f1"]), "val_auc": max(history["val_auc"])})
        all_val_auc.append(history["val_auc"])

    # Save fold summary
    fold_summary_path = os.path.join(output_dir, "fold_summary.csv")
    pd.DataFrame(fold_results).to_csv(fold_summary_path, index=False)
    print(f"Saved fold summary at {fold_summary_path}")

    # Save all_val_auc list to file
    with open(os.path.join(output_dir, "all_val_auc.json"), "w") as f:
        json.dump(all_val_auc, f)
    print(f"Saved all_val_auc at {os.path.join(output_dir, 'all_val_auc.json')}")

# Call the function for each type
train_and_evaluate(file_paths1, output_dir1)
train_and_evaluate(file_paths2, output_dir2)
train_and_evaluate(file_paths3, output_dir3)
train_and_evaluate(file_paths4, output_dir4)
