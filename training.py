import torch
import torch.optim as optim
import time
import wandb  # Weights and Biases for tracking
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from architecture import CNNModel
import torch.nn as nn
import os
import zipfile

def train_model(target_size=(150, 150), batch_size=32, epochs=10, use_gpu=True, data_path='', model_path=''):
    """
    Train a CNN model for dog vs cat classification using PyTorch with Weights & Biases tracking.

    Args:
        target_size (tuple): The target size for the images.
        batch_size (int): Number of images per batch.
        epochs (int): Number of training epochs.
    """

    print('***')
    # Initialize Weights & Biases
    wandb.init(project="dog-cat-classification", config={
        "target_size": target_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "use_gpu": use_gpu
    })

    print(f"Checking dataset at: {data_path}")

    if not os.path.exists(data_path):
        print("Available directories:", os.listdir("/"))
        raise ValueError(f"Data directory '{data_path}' does not exist!")

    # Extract dataset if it's in a ZIP file
    zip_path = os.path.join(data_path, "cats_vs_dogs.zip")
    if os.path.exists(zip_path):
        print(f"Extracting dataset from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        print("Dataset extracted successfully!")

    train_dir = os.path.join(data_path, "train")
    val_dir = os.path.join(data_path, "val")

    # Define data transformations
    target_size = tuple(target_size)
    transform = transforms.Compose([
        transforms.Resize(tuple(target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    if use_gpu and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
    else:
        print("Using CPU")

    model = CNNModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Track model with Weights & Biases
    wandb.watch(model, log="all")

    # Training loop with elapsed time calculation
    start_time = time.time()
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation phase
        model.eval()
        running_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_loss = running_loss / len(val_loader)
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Log training and validation metrics to Weights & Biases
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc
        })

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")

    model_path = os.path.join(model_path, 'cnn_dog_cat_classifier.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model training completed and saved at {model_path}")

    # Finish Weights & Biases run
    wandb.finish()

    return epochs, train_losses, val_losses, train_accuracies, val_accuracies